# ============================================================
# train_rl_agent.py
#
# END-TO-END VISUAL FOLLOW RL
# V16.4
#
# MuJoCo + Gymnasium + Stable-Baselines3 SAC
#
# V16.4
# ------------------------------------------------------------
# Main changes from V16.3:
#
#   1. Removed privileged target geometry from observation.
#   2. Observation state reduced from 16 -> 11.
#   3. True distance / bearing / elevation remain internal
#      for reward and diagnostics only.
#   4. Evaluation environment follows the current curriculum
#      level of the training environment.
#   5. Curriculum changes only between episodes.
#   6. Added detailed curriculum / episode statistics.
#   7. Added visible_ratio / follow_ratio / mean_distance.
#   8. Added max_stable_steps diagnostic.
#   9. Added episode curriculum level diagnostics.
#  10. SAC architecture and core hyperparameters preserved.
#
# Observation:
#   image = 48 x 48 x 3
#   state = 11
#
# State:
#   0  visual_dx
#   1  visual_dy
#   2  visible
#   3  roll
#   4  pitch
#   5  yaw sin
#   6  yaw cos
#   7  altitude
#   8  vx
#   9  vy
#  10  vz
#
# IMPORTANT:
#
#   distance
#   bearing
#   elevation
#   horizontal_distance
#
# are NOT exposed to the agent.
#
# They are still available internally for reward and diagnostics.
#
# Action:
#   [roll, pitch, yaw, throttle]
#   all normalized to [-1, +1]
#
# ============================================================

import os
import math
import argparse
from dataclasses import dataclass

import numpy as np
import mujoco
import gymnasium as gym

from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor


# ============================================================
# VERSION
# ============================================================

VERSION = "V16.4"

print("=" * 76)
print("END-TO-END VISUAL FOLLOW V16.4")
print("=" * 76)


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

GRAVITY = 9.81

RC_MIN = 1000
RC_MID = 1500
RC_MAX = 2000

CONTROL_DT = 0.025

DEFAULT_WIDTH = 48
DEFAULT_HEIGHT = 48

MIN_ALTITUDE = 0.08
MAX_ALTITUDE = 50.0

MAX_EPISODE_STEPS = 800


# ============================================================
# FOLLOW GEOMETRY
# ============================================================

TARGET_FOLLOW_DISTANCE = 15.0

FOLLOW_MIN_DISTANCE = 12.0
FOLLOW_MAX_DISTANCE = 18.0

TOO_CLOSE_DISTANCE = 3.0
HARD_TOO_CLOSE_DISTANCE = 2.0

MAX_VISUAL_DISTANCE = 40.0


# ============================================================
# TARGET SPEED
# ============================================================

TARGET_SPEED_MIN = 4.0
TARGET_SPEED_MAX = 8.0


# ============================================================
# CAMERA
# ============================================================

FOV_H_DEG = 100.0
FOV_V_DEG = 70.0

FOV_H = math.radians(FOV_H_DEG)
FOV_V = math.radians(FOV_V_DEG)


# ============================================================
# ALTITUDE
# ============================================================

DEFAULT_TARGET_ALTITUDE = 3.0

ALTITUDE_TOLERANCE = 0.35
VERTICAL_SPEED_TOLERANCE = 0.50


# ============================================================
# CURRICULUM
#
# 0 = stabilization / altitude
# 1 = visual acquisition
# 2 = distance hold
# 3 = moving target
# ============================================================

CURRICULUM_MAX_LEVEL = 3

PHASE0_MIN_STEPS = 160
PHASE1_MIN_STEPS = 200
PHASE2_MIN_STEPS = 250

CURRICULUM_VISIBLE_THRESHOLD = 0.50
CURRICULUM_FOLLOW_THRESHOLD = 0.20


# ============================================================
# UTILITY
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def normalize(value, low, high):

    if high <= low:
        return 0.0

    return clamp(
        (float(value) - low) / (high - low),
        0.0,
        1.0,
    )


def denormalize(value, low, high):

    return (
        low
        + (float(value) + 1.0)
        * 0.5
        * (high - low)
    )


def safe_float(value, default=0.0):

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:

        return default


def smooth_quality(error, scale):

    scale = max(float(scale), 1e-6)

    return math.exp(
        -abs(float(error)) / scale
    )


# ============================================================
# FC CONFIG
# ============================================================

@dataclass
class FCConfig:

    # --------------------------------------------------------
    # Vertical
    # --------------------------------------------------------

    altitude_kp: float = 1.6

    velocity_kp: float = 0.75

    velocity_kd: float = 0.20

    max_vertical_speed: float = 3.0

    max_vertical_acceleration: float = 1.5

    # --------------------------------------------------------
    # Attitude
    # --------------------------------------------------------

    max_roll_angle: float = math.radians(30.0)

    max_pitch_angle: float = math.radians(30.0)

    max_yaw_rate: float = math.radians(90.0)

    attitude_kp: float = 3.0

    yaw_kp: float = 1.5

    # --------------------------------------------------------
    # Horizontal
    # --------------------------------------------------------

    max_horizontal_acceleration: float = 4.0

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    min_altitude: float = 0.08

    max_altitude: float = 50.0


# ============================================================
# VIRTUAL MAVLINK
# ============================================================

class VirtualMAVLink:

    def __init__(self):

        self.connected = False

        self.rc_channels = np.full(
            18,
            RC_MID,
            dtype=np.int32,
        )

        self.last_command = None
        self.last_command_params = None

    def connect(self):

        self.connected = True

        print(
            "[MAVLINK] Virtual MAVLink connected."
        )

        return True

    def close(self):

        self.connected = False

    def rc_channels_override(
        self,
        channels,
    ):

        if not self.connected:
            return

        for i, value in enumerate(channels):

            if i >= len(self.rc_channels):
                break

            if value is None:
                continue

            self.rc_channels[i] = int(
                clamp(
                    int(value),
                    RC_MIN,
                    RC_MAX,
                )
            )

    def get_rc(
        self,
        channel,
    ):

        index = int(channel) - 1

        if index < 0:
            return RC_MID

        if index >= len(self.rc_channels):
            return RC_MID

        return int(
            self.rc_channels[index]
        )

    def command_long(
        self,
        command,
        *params,
    ):

        self.last_command = command
        self.last_command_params = params


# ============================================================
# VIRTUAL FLIGHT CONTROLLER
# ============================================================

class VirtualFlightController:

    MODE_RAW = "RAW"
    MODE_ALT_HOLD = "ALT_HOLD"

    def __init__(
        self,
        mass,
        motor_min,
        motor_max,
        config=None,
    ):

        self.mass = float(mass)

        self.weight = (
            self.mass
            * GRAVITY
        )

        self.motor_min = float(
            motor_min
        )

        self.motor_max = float(
            motor_max
        )

        self.config = (
            config
            if config is not None
            else FCConfig()
        )

        self.hover_motor_thrust = (
            self.weight / 4.0
        )

        self.hover_throttle = clamp(
            self.hover_motor_thrust
            / max(self.motor_max, 1e-6),
            0.0,
            1.0,
        )

        self.mode = (
            self.MODE_ALT_HOLD
        )

        self.target_altitude = (
            DEFAULT_TARGET_ALTITUDE
        )

        self.last_throttle = (
            self.hover_throttle
        )

        self.last_vertical_acceleration = 0.0

        self.last_roll = 0.0
        self.last_pitch = 0.0
        self.last_yaw = 0.0

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    def set_mode(
        self,
        mode,
        current_altitude=None,
    ):

        if mode not in (
            self.MODE_RAW,
            self.MODE_ALT_HOLD,
        ):

            raise ValueError(
                f"Unknown FC mode: {mode}"
            )

        self.mode = mode

        if current_altitude is not None:

            self.target_altitude = float(
                current_altitude
            )

    # --------------------------------------------------------
    # Target altitude
    # --------------------------------------------------------

    def set_target_altitude(
        self,
        altitude,
    ):

        self.target_altitude = clamp(
            altitude,
            self.config.min_altitude,
            self.config.max_altitude,
        )

    # --------------------------------------------------------
    # Collective
    # --------------------------------------------------------

    def compute_collective(
        self,
        throttle_command,
        altitude,
        vertical_velocity,
        dt,
    ):

        throttle_command = clamp(
            throttle_command,
            -1.0,
            1.0,
        )

        requested_acceleration = (
            throttle_command
            * self.config.max_vertical_acceleration
        )

        damping = (
            -0.20
            * vertical_velocity
        )

        acceleration = (
            requested_acceleration
            + damping
        )

        acceleration = clamp(
            acceleration,
            -self.config.max_vertical_acceleration,
            self.config.max_vertical_acceleration,
        )

        self.last_vertical_acceleration = (
            acceleration
        )

        total_thrust = (
            self.mass
            * (
                GRAVITY
                + acceleration
            )
        )

        motor_thrust = (
            total_thrust / 4.0
        )

        motor_thrust = clamp(
            motor_thrust,
            self.motor_min,
            self.motor_max,
        )

        throttle = (
            motor_thrust
            / max(self.motor_max, 1e-6)
        )

        self.last_throttle = (
            throttle
        )

        return float(throttle)

    # --------------------------------------------------------
    # Motor mixer
    # --------------------------------------------------------

    def mix_motors(
        self,
        collective,
        roll,
        pitch,
        yaw_rate,
    ):

        collective = clamp(
            float(collective),
            0.0,
            1.0,
        )

        roll = clamp(
            float(roll),
            -1.0,
            1.0,
        )

        pitch = clamp(
            float(pitch),
            -1.0,
            1.0,
        )

        yaw_rate = clamp(
            float(yaw_rate),
            -1.0,
            1.0,
        )

        roll_mix = (
            0.12 * roll
        )

        pitch_mix = (
            0.12 * pitch
        )

        yaw_mix = (
            0.06 * yaw_rate
        )

        m1 = (
            collective
            + roll_mix
            - pitch_mix
            + yaw_mix
        )

        m2 = (
            collective
            - roll_mix
            - pitch_mix
            - yaw_mix
        )

        m3 = (
            collective
            - roll_mix
            + pitch_mix
            + yaw_mix
        )

        m4 = (
            collective
            + roll_mix
            + pitch_mix
            - yaw_mix
        )

        return np.clip(
            np.array(
                [
                    m1,
                    m2,
                    m3,
                    m4,
                ],
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )

    # --------------------------------------------------------
    # Complete motor computation
    # --------------------------------------------------------

    def compute_motors(
        self,
        throttle_command,
        roll_command,
        pitch_command,
        yaw_command,
        altitude,
        vertical_velocity,
        dt,
    ):

        collective = (
            self.compute_collective(
                throttle_command,
                altitude,
                vertical_velocity,
                dt,
            )
        )

        motors_normalized = (
            self.mix_motors(
                collective,
                roll_command,
                pitch_command,
                yaw_command,
            )
        )

        motor_thrust = (
            self.motor_min
            + motors_normalized
            * (
                self.motor_max
                - self.motor_min
            )
        )

        self.last_roll = float(
            roll_command
        )

        self.last_pitch = float(
            pitch_command
        )

        self.last_yaw = float(
            yaw_command
        )

        return motor_thrust


# ============================================================
# ENVIRONMENT
# ============================================================

class RealisticVisualFollowEnv(
    gym.Env
):

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 40,
    }

    def __init__(
        self,
        xml_path="scene.xml",
        render_width=48,
        render_height=48,
        diagnostic=False,
        curriculum=True,
    ):

        super().__init__()

        self.xml_path = xml_path

        self.render_width = int(
            render_width
        )

        self.render_height = int(
            render_height
        )

        self.diagnostic = diagnostic

        self.curriculum_enabled = bool(
            curriculum
        )

        print()
        print("=" * 76)
        print(
            "CREATING END-TO-END VISUAL FOLLOW "
            "ENVIRONMENT V16.4"
        )
        print("=" * 76)

        # ====================================================
        # MODEL
        # ====================================================

        print(
            "[ENV] Loading MuJoCo model..."
        )

        self.model = (
            mujoco.MjModel.from_xml_path(
                self.xml_path
            )
        )

        self.data = mujoco.MjData(
            self.model
        )

        print(
            f"[ENV] Model loaded: "
            f"nbody={self.model.nbody}, "
            f"njnt={self.model.njnt}, "
            f"nu={self.model.nu}"
        )

        # ====================================================
        # DRONE
        # ====================================================

        self.drone_body_id = (
            self._find_body("x2")
        )

        print(
            f"[ENV] Body 'x2' -> id "
            f"{self.drone_body_id}"
        )

        # ====================================================
        # TARGET
        # ====================================================

        self.target_body_id = (
            self._find_body(
                "target_kral"
            )
        )

        print(
            f"[ENV] Body 'target_kral' -> id "
            f"{self.target_body_id}"
        )

        self.target_mocap_id = int(
            self.model.body_mocapid[
                self.target_body_id
            ]
        )

        if self.target_mocap_id >= 0:

            print(
                "[ENV] Target is MOCAP."
            )

            print(
                f"[ENV] mocap id: "
                f"{self.target_mocap_id}"
            )

        else:

            print(
                "[ENV] Target is NOT MOCAP."
            )

        # ====================================================
        # DRONE FREE JOINT
        # ====================================================

        self.drone_joint_id = -1

        for j in range(
            self.model.njnt
        ):

            body_id = int(
                self.model.jnt_bodyid[j]
            )

            if body_id != (
                self.drone_body_id
            ):
                continue

            joint_type = (
                self.model.jnt_type[j]
            )

            if (
                joint_type
                == mujoco.mjtJoint.mjJNT_FREE
            ):

                self.drone_joint_id = j
                break

        if self.drone_joint_id < 0:

            raise RuntimeError(
                "Drone free joint not found."
            )

        self.drone_qpos_adr = int(
            self.model.jnt_qposadr[
                self.drone_joint_id
            ]
        )

        self.drone_qvel_adr = int(
            self.model.jnt_dofadr[
                self.drone_joint_id
            ]
        )

        # ====================================================
        # MASS
        # ====================================================

        self.drone_mass = (
            self._calculate_drone_mass()
        )

        print(
            f"[ENV] Drone mass: "
            f"{self.drone_mass:.4f} kg"
        )

        # ====================================================
        # MOTORS
        # ====================================================

        print(
            "[ENV] Searching for motor actuators..."
        )

        self.motor_actuators = []

        wanted = {
            "thrust1",
            "thrust2",
            "thrust3",
            "thrust4",
        }

        for actuator_id in range(
            self.model.nu
        ):

            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )

            print(
                f"[ENV] actuator "
                f"{actuator_id}: {name}"
            )

            if name in wanted:

                self.motor_actuators.append(
                    actuator_id
                )

        if len(
            self.motor_actuators
        ) != 4:

            raise RuntimeError(
                "Expected exactly four motor actuators."
            )

        self.motor_actuators.sort()

        self.motor_ctrl_min = (
            np.zeros(
                4,
                dtype=np.float64,
            )
        )

        self.motor_ctrl_max = (
            np.zeros(
                4,
                dtype=np.float64,
            )
        )

        for i, actuator_id in enumerate(
            self.motor_actuators
        ):

            self.motor_ctrl_min[i] = (
                self.model.actuator_ctrlrange[
                    actuator_id,
                    0,
                ]
            )

            self.motor_ctrl_max[i] = (
                self.model.actuator_ctrlrange[
                    actuator_id,
                    1,
                ]
            )

        self.motor_min_thrust = float(
            np.max(
                self.motor_ctrl_min
            )
        )

        self.motor_max_thrust = float(
            np.min(
                self.motor_ctrl_max
            )
        )

        if (
            self.motor_max_thrust
            <= self.motor_min_thrust
        ):

            raise RuntimeError(
                "Invalid motor control range."
            )

        # ====================================================
        # PHYSICS
        # ====================================================

        self.weight = (
            self.drone_mass
            * GRAVITY
        )

        self.hover_force = (
            self.weight / 4.0
        )

        self.hover_throttle = clamp(
            self.hover_force
            / self.motor_max_thrust,
            0.0,
            1.0,
        )

        print(
            f"[ENV] Weight: "
            f"{self.weight:.4f} N"
        )

        print(
            f"[ENV] Hover force/motor: "
            f"{self.hover_force:.4f} N"
        )

        print(
            f"[ENV] Hover throttle: "
            f"{self.hover_throttle:.6f}"
        )

        # ====================================================
        # FC
        # ====================================================

        self.fc = (
            VirtualFlightController(
                mass=self.drone_mass,
                motor_min=self.motor_min_thrust,
                motor_max=self.motor_max_thrust,
                config=FCConfig(),
            )
        )

        # ====================================================
        # MAVLINK
        # ====================================================

        self.mavlink = (
            VirtualMAVLink()
        )

        self.mavlink.connect()

        # ====================================================
        # CAMERA
        # ====================================================

        self.camera_id = (
            self._find_camera(
                "front_camera"
            )
        )

        print(
            f"[ENV] Camera 'front_camera' -> id "
            f"{self.camera_id}"
        )

        # ====================================================
        # RENDERER
        # ====================================================

        self.renderer = None

        self._init_renderer()

        # ====================================================
        # ACTION SPACE
        # ====================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # ====================================================
        # OBSERVATION
        # ====================================================
        #
        # V16.4 VISUAL-ONLY STATE
        #
        # 0  visual_dx
        # 1  visual_dy
        # 2  visible
        # 3  roll
        # 4  pitch
        # 5  yaw sin
        # 6  yaw cos
        # 7  altitude
        # 8  vx
        # 9  vy
        # 10 vz
        #
        # PRIVILEGED TARGET GEOMETRY REMOVED:
        #
        # distance
        # bearing
        # elevation
        # horizontal_distance
        #
        # ====================================================

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.render_height,
                        self.render_width,
                        3,
                    ),
                    dtype=np.uint8,
                ),

                "state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(11,),
                    dtype=np.float32,
                ),
            }
        )

        # ====================================================
        # RUNTIME
        # ====================================================

        self.step_count = 0

        self.target_velocity = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

        self.target_distance = (
            TARGET_FOLLOW_DISTANCE
        )

        self.last_action = (
            np.zeros(
                4,
                dtype=np.float32,
            )
        )

        self.previous_action = (
            np.zeros(
                4,
                dtype=np.float32,
            )
        )

        self.last_rc = np.full(
            18,
            RC_MID,
            dtype=np.int32,
        )

        self.last_reason = ""

        self.episode_reward = 0.0

        self.episode_distance_sum = 0.0

        self.episode_visible_steps = 0

        self.episode_follow_steps = 0

        self.episode_stable_steps = 0

        self.episode_max_stable_steps = 0

        self.episode_success = False

        self.success_hold_steps = 0

        self.episode_level = 0

        # ----------------------------------------------------
        # Previous errors for progress reward
        # ----------------------------------------------------

        self.prev_center_error = 1.0

        self.prev_distance_error = (
            TARGET_FOLLOW_DISTANCE
        )

        self.prev_altitude_error = 1.0

        self.prev_visible = False

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        self.reward_components = {}

        # ----------------------------------------------------
        # Curriculum
        # ----------------------------------------------------

        self.curriculum_level = 0

        print(
            "[ENV] Observation state: 11"
        )

        print(
            "[ENV] Privileged target geometry: REMOVED"
        )

        print(
            "[ENV] Initialization complete."
        )

        print("=" * 76)

    # ========================================================
    # FIND BODY
    # ========================================================

    def _find_body(
        self,
        name,
    ):

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )

        if body_id < 0:

            raise RuntimeError(
                f"MuJoCo body '{name}' not found."
            )

        return int(body_id)

    # ========================================================
    # FIND CAMERA
    # ========================================================

    def _find_camera(
        self,
        name,
    ):

        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            name,
        )

        if camera_id < 0:

            raise RuntimeError(
                f"MuJoCo camera '{name}' not found."
            )

        return int(camera_id)

    # ========================================================
    # MASS
    # ========================================================

    def _calculate_drone_mass(
        self,
    ):

        mass = 0.0

        for body_id in range(
            self.model.nbody
        ):

            current = body_id
            belongs_to_drone = False

            while current != 0:

                if (
                    current
                    == self.drone_body_id
                ):

                    belongs_to_drone = True
                    break

                current = int(
                    self.model.body_parentid[
                        current
                    ]
                )

            if belongs_to_drone:

                mass += float(
                    self.model.body_mass[
                        body_id
                    ]
                )

        if mass <= 0.0:

            mass = float(
                self.model.body_subtreemass[
                    self.drone_body_id
                ]
            )

        return float(mass)

    # ========================================================
    # RENDERER
    # ========================================================

    def _init_renderer(
        self,
    ):

        try:

            self.renderer = (
                mujoco.Renderer(
                    self.model,
                    height=self.render_height,
                    width=self.render_width,
                )
            )

            print(
                f"[RENDER] Renderer OK: "
                f"{self.render_width}x"
                f"{self.render_height}"
            )

        except Exception as e:

            print(
                f"[RENDER] Renderer ERROR: {e}"
            )

            self.renderer = None

    # ========================================================
    # DRONE POSITION
    # ========================================================

    def _get_drone_position(
        self,
    ):

        return np.asarray(
            self.data.qpos[
                self.drone_qpos_adr:
                self.drone_qpos_adr + 3
            ],
            dtype=np.float64,
        ).copy()

    # ========================================================
    # DRONE VELOCITY
    # ========================================================

    def _get_drone_velocity(
        self,
    ):

        return np.asarray(
            self.data.qvel[
                self.drone_qvel_adr:
                self.drone_qvel_adr + 3
            ],
            dtype=np.float64,
        ).copy()

    # ========================================================
    # ANGULAR VELOCITY
    # ========================================================

    def _get_angular_velocity(
        self,
    ):

        return np.asarray(
            self.data.qvel[
                self.drone_qvel_adr + 3:
                self.drone_qvel_adr + 6
            ],
            dtype=np.float64,
        ).copy()

    # ========================================================
    # TARGET POSITION
    # ========================================================

    def _get_target_position(
        self,
    ):

        if self.target_mocap_id >= 0:

            return np.asarray(
                self.data.mocap_pos[
                    self.target_mocap_id
                ],
                dtype=np.float64,
            ).copy()

        return np.asarray(
            self.data.xpos[
                self.target_body_id
            ],
            dtype=np.float64,
        ).copy()

    # ========================================================
    # TARGET UPDATE
    # ========================================================

    def _update_target(
        self,
        dt,
    ):

        if self.target_mocap_id < 0:
            return

        self.data.mocap_pos[
            self.target_mocap_id
        ] += (
            self.target_velocity
            * dt
        )

        pos = self.data.mocap_pos[
            self.target_mocap_id
        ]

        # ----------------------------------------------------
        # Horizontal boundaries
        # ----------------------------------------------------

        if abs(pos[0]) > 100.0:

            self.target_velocity[0] *= -1.0

            pos[0] = clamp(
                pos[0],
                -100.0,
                100.0,
            )

        if abs(pos[1]) > 100.0:

            self.target_velocity[1] *= -1.0

            pos[1] = clamp(
                pos[1],
                -100.0,
                100.0,
            )

        # ----------------------------------------------------
        # Vertical boundaries
        # ----------------------------------------------------

        if pos[2] < 3.0:

            self.target_velocity[2] = abs(
                self.target_velocity[2]
            )

        elif pos[2] > 40.0:

            self.target_velocity[2] = -abs(
                self.target_velocity[2]
            )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # ====================================================
        # DRONE INITIAL STATE
        # ====================================================

        start_altitude = 0.30

        self.data.qpos[
            self.drone_qpos_adr:
            self.drone_qpos_adr + 3
        ] = np.array(
            [
                0.0,
                0.0,
                start_altitude,
            ],
            dtype=np.float64,
        )

        self.data.qpos[
            self.drone_qpos_adr + 3:
            self.drone_qpos_adr + 7
        ] = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        self.data.qvel[
            self.drone_qvel_adr:
            self.drone_qvel_adr + 6
        ] = 0.0

        # ====================================================
        # CURRICULUM LEVEL
        # ====================================================

        if self.curriculum_enabled:

            level = int(
                self.curriculum_level
            )

        else:

            level = 3

        self.episode_level = level

        # ----------------------------------------------------
        # Level 0
        # Stable target close to center
        # ----------------------------------------------------

        if level <= 0:

            distance = self.np_random.uniform(
                18.0,
                20.0,
            )

            elevation = self.np_random.uniform(
                math.radians(8.0),
                math.radians(15.0),
            )

            target_speed = 0.0

        # ----------------------------------------------------
        # Level 1
        # Visual acquisition
        # ----------------------------------------------------

        elif level == 1:

            distance = self.np_random.uniform(
                16.0,
                22.0,
            )

            elevation = self.np_random.uniform(
                math.radians(8.0),
                math.radians(22.0),
            )

            target_speed = 0.0

        # ----------------------------------------------------
        # Level 2
        # Distance hold
        # ----------------------------------------------------

        elif level == 2:

            distance = self.np_random.uniform(
                14.0,
                22.0,
            )

            elevation = self.np_random.uniform(
                math.radians(8.0),
                math.radians(30.0),
            )

            target_speed = self.np_random.uniform(
                1.0,
                3.0,
            )

        # ----------------------------------------------------
        # Level 3
        # Full moving target
        # ----------------------------------------------------

        else:

            distance = self.np_random.uniform(
                12.0,
                25.0,
            )

            elevation = self.np_random.uniform(
                math.radians(10.0),
                math.radians(45.0),
            )

            target_speed = self.np_random.uniform(
                TARGET_SPEED_MIN,
                TARGET_SPEED_MAX,
            )

        # ====================================================
        # TARGET POSITION
        # ====================================================

        azimuth = self.np_random.uniform(
            -math.pi,
            math.pi,
        )

        horizontal_distance = (
            distance
            * math.cos(elevation)
        )

        target_x = (
            horizontal_distance
            * math.cos(azimuth)
        )

        target_y = (
            horizontal_distance
            * math.sin(azimuth)
        )

        target_z = (
            DEFAULT_TARGET_ALTITUDE
            + distance
            * math.sin(elevation)
        )

        target_z = clamp(
            target_z,
            3.0,
            35.0,
        )

        if self.target_mocap_id >= 0:

            self.data.mocap_pos[
                self.target_mocap_id
            ] = np.array(
                [
                    target_x,
                    target_y,
                    target_z,
                ],
                dtype=np.float64,
            )

            self.data.mocap_quat[
                self.target_mocap_id
            ] = np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        self.target_distance = float(
            distance
        )

        # ====================================================
        # TARGET VELOCITY
        # ====================================================

        target_heading = self.np_random.uniform(
            -math.pi,
            math.pi,
        )

        self.target_velocity = np.array(
            [
                target_speed
                * math.cos(target_heading),

                target_speed
                * math.sin(target_heading),

                0.0,
            ],
            dtype=np.float64,
        )

        # ====================================================
        # FC
        # ====================================================

        self.fc.set_mode(
            VirtualFlightController.MODE_ALT_HOLD,
            current_altitude=start_altitude,
        )

        self.fc.set_target_altitude(
            DEFAULT_TARGET_ALTITUDE
        )

        # ====================================================
        # RUNTIME RESET
        # ====================================================

        self.step_count = 0

        self.last_action[:] = 0.0

        self.previous_action[:] = 0.0

        self.last_rc[:] = RC_MID

        self.last_reason = ""

        self.episode_reward = 0.0

        self.episode_distance_sum = 0.0

        self.episode_visible_steps = 0

        self.episode_follow_steps = 0

        self.episode_stable_steps = 0

        self.episode_max_stable_steps = 0

        self.episode_success = False

        self.success_hold_steps = 0

        self.prev_center_error = 1.0

        self.prev_distance_error = (
            abs(
                self.target_distance
                - TARGET_FOLLOW_DISTANCE
            )
        )

        self.prev_altitude_error = (
            abs(
                DEFAULT_TARGET_ALTITUDE
                - start_altitude
            )
        )

        self.prev_visible = False

        self.reward_components = {}

        # ====================================================
        # FORWARD
        # ====================================================

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        observation = (
            self._get_observation()
        )

        info = {

            "curriculum_level":
                self.curriculum_level,

            "episode_level":
                self.episode_level,

            "phase":
                self._get_phase_name(),

            "target_distance":
                self.target_distance,

            "altitude":
                start_altitude,

            "success":
                False,

        }

        return (
            observation,
            info,
        )

    # ========================================================
    # PHASE NAME
    # ========================================================

    def _get_phase_name(
        self,
    ):

        level = (
            3
            if not self.curriculum_enabled
            else self.curriculum_level
        )

        if level <= 0:
            return "stabilization"

        if level == 1:
            return "visual_acquisition"

        if level == 2:
            return "distance_hold"

        return "moving_follow"

    # ========================================================
    # PHASE NAME FROM EPISODE LEVEL
    # ========================================================

    def _get_episode_phase_name(
        self,
    ):

        level = int(
            self.episode_level
        )

        if level <= 0:
            return "stabilization"

        if level == 1:
            return "visual_acquisition"

        if level == 2:
            return "distance_hold"

        return "moving_follow"

    # ========================================================
    # RC
    # ========================================================

    def _send_rc_override(
        self,
        throttle,
        roll=1500,
        pitch=1500,
        yaw=1500,
    ):

        channels = [
            int(roll),
            int(pitch),
            int(throttle),
            int(yaw),
        ]

        self.last_rc[:4] = channels

        self.mavlink.rc_channels_override(
            channels
        )

    # ========================================================
    # ACTION -> RC
    # ========================================================

    def _action_to_rc(
        self,
        action,
    ):

        roll = float(action[0])
        pitch = float(action[1])
        yaw = float(action[2])
        throttle = float(action[3])

        roll_rc = int(
            round(
                denormalize(
                    roll,
                    1000,
                    2000,
                )
            )
        )

        pitch_rc = int(
            round(
                denormalize(
                    pitch,
                    1000,
                    2000,
                )
            )
        )

        yaw_rc = int(
            round(
                denormalize(
                    yaw,
                    1000,
                    2000,
                )
            )
        )

        throttle_rc = int(
            round(
                1500.0
                + throttle
                * 500.0
            )
        )

        return (
            roll_rc,
            pitch_rc,
            yaw_rc,
            throttle_rc,
        )

    # ========================================================
    # APPLY MOTORS
    # ========================================================

    def _apply_motors(
        self,
        motors,
    ):

        for i, actuator_id in enumerate(
            self.motor_actuators
        ):

            value = clamp(
                float(motors[i]),
                self.motor_ctrl_min[i],
                self.motor_ctrl_max[i],
            )

            self.data.ctrl[
                actuator_id
            ] = value

    # ========================================================
    # QUATERNION -> EULER
    # ========================================================

    def _get_euler(
        self,
    ):

        quat = np.asarray(
            self.data.qpos[
                self.drone_qpos_adr + 3:
                self.drone_qpos_adr + 7
            ],
            dtype=np.float64,
        )

        w, x, y, z = quat

        sinr = (
            2.0
            * (
                w * x
                + y * z
            )
        )

        cosr = (
            1.0
            - 2.0
            * (
                x * x
                + y * y
            )
        )

        roll = math.atan2(
            sinr,
            cosr,
        )

        sinp = (
            2.0
            * (
                w * y
                - z * x
            )
        )

        pitch = math.asin(
            clamp(
                sinp,
                -1.0,
                1.0,
            )
        )

        siny = (
            2.0
            * (
                w * z
                + x * y
            )
        )

        cosy = (
            1.0
            - 2.0
            * (
                y * y
                + z * z
            )
        )

        yaw = math.atan2(
            siny,
            cosy,
        )

        return (
            float(roll),
            float(pitch),
            float(yaw),
        )

    # ========================================================
    # CAMERA DETECTION
    # ========================================================

    def _calculate_visual_detection(
        self,
    ):

        drone = (
            self._get_drone_position()
        )

        target = (
            self._get_target_position()
        )

        delta = (
            target - drone
        )

        distance = float(
            np.linalg.norm(delta)
        )

        if distance < 1e-6:

            return (
                0.0,
                0.0,
                0.0,
                1.0,
                True,
            )

        quat = np.asarray(
            self.data.qpos[
                self.drone_qpos_adr + 3:
                self.drone_qpos_adr + 7
            ],
            dtype=np.float64,
        )

        rotation = np.zeros(
            (3, 3),
            dtype=np.float64,
        )

        mujoco.mju_quat2Mat(
            rotation.reshape(-1),
            quat,
        )

        forward = rotation[:, 0]
        right = rotation[:, 1]
        up = rotation[:, 2]

        direction = (
            delta / distance
        )

        forward_dot = float(
            np.dot(
                direction,
                forward,
            )
        )

        horizontal_angle = math.atan2(
            float(
                np.dot(
                    direction,
                    right,
                )
            ),
            forward_dot,
        )

        vertical_angle = math.atan2(
            float(
                np.dot(
                    direction,
                    up,
                )
            ),
            forward_dot,
        )

        visible = (
            forward_dot > 0.0
            and abs(horizontal_angle)
            <= FOV_H * 0.5
            and abs(vertical_angle)
            <= FOV_V * 0.5
            and distance
            <= MAX_VISUAL_DISTANCE
        )

        dx = clamp(
            horizontal_angle
            / (FOV_H * 0.5),
            -1.0,
            1.0,
        )

        dy = clamp(
            vertical_angle
            / (FOV_V * 0.5),
            -1.0,
            1.0,
        )

        normalized_distance = clamp(
            distance
            / MAX_VISUAL_DISTANCE,
            0.0,
            1.0,
        )

        return (
            float(dx),
            float(dy),
            float(distance),
            float(normalized_distance),
            bool(visible),
        )

    # ========================================================
    # RENDER CAMERA
    # ========================================================

    def _render_camera(
        self,
    ):

        if self.renderer is None:

            return np.zeros(
                (
                    self.render_height,
                    self.render_width,
                    3,
                ),
                dtype=np.uint8,
            )

        try:

            self.renderer.update_scene(
                self.data,
                camera=self.camera_id,
            )

            image = (
                self.renderer.render()
            )

            image = np.asarray(
                image,
                dtype=np.uint8,
            )

            if (
                image.ndim != 3
                or image.shape[-1] != 3
            ):

                raise RuntimeError(
                    "Renderer returned invalid RGB image."
                )

            return image.copy()

        except Exception as e:

            if self.diagnostic:

                print(
                    f"[RENDER] Error: {e}"
                )

            return np.zeros(
                (
                    self.render_height,
                    self.render_width,
                    3,
                ),
                dtype=np.uint8,
            )

    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_observation(
        self,
    ):

        drone = (
            self._get_drone_position()
        )

        velocity = (
            self._get_drone_velocity()
        )

        (
            visual_dx,
            visual_dy,
            _distance,
            _normalized_distance,
            visible,
        ) = (
            self._calculate_visual_detection()
        )

        roll, pitch, yaw = (
            self._get_euler()
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # No target distance.
        # No target bearing.
        # No target elevation.
        # No horizontal distance.
        #
        # Only visual information + vehicle state.
        # ----------------------------------------------------

        state = np.array(
            [
                visual_dx,

                visual_dy,

                1.0
                if visible
                else -1.0,

                clamp(
                    roll
                    / math.radians(45.0),
                    -1.0,
                    1.0,
                ),

                clamp(
                    pitch
                    / math.radians(45.0),
                    -1.0,
                    1.0,
                ),

                math.sin(yaw),

                math.cos(yaw),

                clamp(
                    drone[2]
                    / MAX_ALTITUDE,
                    0.0,
                    1.0,
                )
                * 2.0
                - 1.0,

                clamp(
                    velocity[0] / 10.0,
                    -1.0,
                    1.0,
                ),

                clamp(
                    velocity[1] / 10.0,
                    -1.0,
                    1.0,
                ),

                clamp(
                    velocity[2] / 5.0,
                    -1.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

        image = (
            self._render_camera()
        )

        return {
            "image": image,
            "state": state,
        }

    # ========================================================
    # REWARD
    # ========================================================

    def _calculate_reward(
        self,
    ):

        (
            dx,
            dy,
            distance,
            _,
            visible,
        ) = (
            self._calculate_visual_detection()
        )

        position = (
            self._get_drone_position()
        )

        velocity = (
            self._get_drone_velocity()
        )

        roll, pitch, _ = (
            self._get_euler()
        )

        altitude = float(
            position[2]
        )

        # ====================================================
        # Current errors
        # ====================================================

        center_error = clamp(
            math.sqrt(
                dx * dx
                + dy * dy
            ),
            0.0,
            1.414,
        )

        distance_error = abs(
            distance
            - TARGET_FOLLOW_DISTANCE
        )

        altitude_error = abs(
            altitude
            - DEFAULT_TARGET_ALTITUDE
        )

        # ====================================================
        # Phase
        # ====================================================

        level = (
            3
            if not self.curriculum_enabled
            else self.episode_level
        )

        reward = 0.0

        components = {}

        # ====================================================
        # PHASE 0
        # Stabilization / altitude
        # ====================================================

        if level == 0:

            altitude_quality = (
                smooth_quality(
                    altitude_error,
                    1.0,
                )
            )

            vertical_quality = (
                smooth_quality(
                    abs(float(velocity[2])),
                    1.5,
                )
            )

            attitude_error = (
                abs(roll)
                + abs(pitch)
            )

            attitude_quality = (
                smooth_quality(
                    attitude_error,
                    math.radians(20.0),
                )
            )

            altitude_reward = (
                0.70
                * altitude_quality
            )

            vertical_reward = (
                0.30
                * vertical_quality
            )

            attitude_reward = (
                0.20
                * attitude_quality
            )

            reward += (
                altitude_reward
                + vertical_reward
                + attitude_reward
            )

            components[
                "altitude"
            ] = altitude_reward

            components[
                "vertical"
            ] = vertical_reward

            components[
                "attitude"
            ] = attitude_reward

        # ====================================================
        # PHASE 1
        # Visual acquisition
        # ====================================================

        elif level == 1:

            altitude_quality = (
                smooth_quality(
                    altitude_error,
                    1.0,
                )
            )

            reward += (
                0.35
                * altitude_quality
            )

            components[
                "altitude"
            ] = (
                0.35
                * altitude_quality
            )

            if visible:

                self.episode_visible_steps += 1

                center_quality = (
                    1.0
                    - (
                        center_error
                        / math.sqrt(2.0)
                    )
                )

                center_reward = (
                    0.90
                    * center_quality
                )

                reward += center_reward

                components[
                    "visual_center"
                ] = center_reward

                visible_reward = 0.25

                reward += visible_reward

                components[
                    "visible"
                ] = visible_reward

            else:

                components[
                    "visible"
                ] = 0.0

        # ====================================================
        # PHASE 2
        # Acquisition + distance hold
        # ====================================================

        elif level == 2:

            altitude_quality = (
                smooth_quality(
                    altitude_error,
                    1.0,
                )
            )

            reward += (
                0.25
                * altitude_quality
            )

            components[
                "altitude"
            ] = (
                0.25
                * altitude_quality
            )

            if visible:

                self.episode_visible_steps += 1

                center_quality = (
                    1.0
                    - (
                        center_error
                        / math.sqrt(2.0)
                    )
                )

                visual_reward = (
                    0.65
                    * center_quality
                )

                reward += visual_reward

                components[
                    "visual_center"
                ] = visual_reward

                distance_quality = (
                    smooth_quality(
                        distance_error,
                        5.0,
                    )
                )

                distance_reward = (
                    0.65
                    * distance_quality
                )

                reward += distance_reward

                components[
                    "distance"
                ] = distance_reward

                if (
                    FOLLOW_MIN_DISTANCE
                    <= distance
                    <= FOLLOW_MAX_DISTANCE
                ):

                    reward += 0.35

                    components[
                        "follow_band"
                    ] = 0.35

                    self.episode_follow_steps += 1

        # ====================================================
        # PHASE 3
        # Full moving-target follow
        # ====================================================

        else:

            if visible:

                self.episode_visible_steps += 1

                center_quality = (
                    1.0
                    - (
                        center_error
                        / math.sqrt(2.0)
                    )
                )

                center_reward = (
                    0.75
                    * center_quality
                )

                reward += center_reward

                components[
                    "visual_center"
                ] = center_reward

                distance_quality = (
                    smooth_quality(
                        distance_error,
                        4.0,
                    )
                )

                distance_reward = (
                    0.75
                    * distance_quality
                )

                reward += distance_reward

                components[
                    "distance"
                ] = distance_reward

                if (
                    FOLLOW_MIN_DISTANCE
                    <= distance
                    <= FOLLOW_MAX_DISTANCE
                ):

                    reward += 0.50

                    components[
                        "follow_band"
                    ] = 0.50

                    self.episode_follow_steps += 1

            # ------------------------------------------------
            # Outside useful distance
            # ------------------------------------------------

            if distance > 35.0:

                far_penalty = -0.15

                reward += far_penalty

                components[
                    "too_far"
                ] = far_penalty

            # ------------------------------------------------
            # Too close
            # ------------------------------------------------

            if distance < TOO_CLOSE_DISTANCE:

                close_penalty = (
                    -1.0
                    * (
                        TOO_CLOSE_DISTANCE
                        - distance
                    )
                )

                reward += close_penalty

                components[
                    "too_close"
                ] = close_penalty

        # ====================================================
        # ALTITUDE STABILITY
        # ====================================================

        vertical_penalty = (
            -0.05
            * min(
                abs(float(velocity[2])),
                3.0,
            )
        )

        reward += vertical_penalty

        components[
            "vertical_stability"
        ] = vertical_penalty

        # ====================================================
        # ATTITUDE
        # ====================================================

        attitude_penalty = (
            -0.04
            * (
                abs(roll)
                + abs(pitch)
            )
        )

        reward += attitude_penalty

        components[
            "attitude_stability"
        ] = attitude_penalty

        # ====================================================
        # ACTION SMOOTHNESS
        # ====================================================

        action_delta = (
            np.abs(
                self.last_action
                - self.previous_action
            )
        )

        smoothness_penalty = (
            -0.01
            * float(
                np.sum(
                    action_delta
                )
            )
        )

        reward += smoothness_penalty

        components[
            "smoothness"
        ] = smoothness_penalty

        # ====================================================
        # ALTITUDE SAFETY
        # ====================================================

        if altitude < 0.20:

            altitude_penalty = -0.40

            reward += altitude_penalty

            components[
                "low_altitude"
            ] = altitude_penalty

        # ====================================================
        # PROGRESS REWARD
        # ====================================================

        altitude_progress = (
            self.prev_altitude_error
            - altitude_error
        )

        altitude_progress = clamp(
            altitude_progress,
            -0.20,
            0.20,
        )

        altitude_progress_reward = (
            0.60
            * altitude_progress
        )

        reward += (
            altitude_progress_reward
        )

        components[
            "altitude_progress"
        ] = altitude_progress_reward

        # ----------------------------------------------------
        # Visual progress
        # ----------------------------------------------------

        if visible:

            center_progress = (
                self.prev_center_error
                - center_error
            )

            center_progress = clamp(
                center_progress,
                -0.20,
                0.20,
            )

            center_progress_reward = (
                0.70
                * center_progress
            )

            reward += (
                center_progress_reward
            )

            components[
                "center_progress"
            ] = center_progress_reward

        # ----------------------------------------------------
        # Distance progress
        # ----------------------------------------------------

        if level >= 2:

            distance_progress = (
                self.prev_distance_error
                - distance_error
            )

            distance_progress = clamp(
                distance_progress,
                -0.50,
                0.50,
            )

            distance_progress_reward = (
                0.40
                * distance_progress
            )

            reward += (
                distance_progress_reward
            )

            components[
                "distance_progress"
            ] = distance_progress_reward

        # ====================================================
        # STABLE FOLLOW
        # ====================================================

        stable_now = (
            visible
            and
            FOLLOW_MIN_DISTANCE
            <= distance
            <= FOLLOW_MAX_DISTANCE
            and
            center_error < 0.30
            and
            altitude_error < 0.60
            and
            abs(float(velocity[2]))
            < 1.0
        )

        if stable_now:

            self.episode_stable_steps += 1

            self.episode_max_stable_steps = max(
                self.episode_max_stable_steps,
                self.episode_stable_steps,
            )

        else:

            self.episode_stable_steps = 0

        # ====================================================
        # SUCCESS HOLD
        # ====================================================

        if (
            level >= 2
            and stable_now
        ):

            self.success_hold_steps += 1

        else:

            self.success_hold_steps = 0

        if (
            level >= 2
            and
            self.success_hold_steps >= 40
            and
            not self.episode_success
        ):

            self.episode_success = True

            success_reward = 5.0

            reward += success_reward

            components[
                "success"
            ] = success_reward

        # ====================================================
        # UPDATE PREVIOUS ERRORS
        # ====================================================

        self.prev_center_error = (
            center_error
        )

        self.prev_distance_error = (
            distance_error
        )

        self.prev_altitude_error = (
            altitude_error
        )

        self.prev_visible = (
            visible
        )

        # ====================================================
        # BOUNDED REWARD
        # ====================================================

        reward = clamp(
            reward,
            -5.0,
            5.0,
        )

        self.reward_components = (
            components
        )

        return float(reward)

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # ====================================================
        # IMPORTANT:
        #
        # Keep previous action BEFORE replacing current one.
        # ====================================================

        self.previous_action = (
            self.last_action.copy()
        )

        self.last_action = (
            action.copy()
        )

        # ====================================================
        # ACTION
        # ====================================================

        roll_command = float(
            action[0]
        )

        pitch_command = float(
            action[1]
        )

        yaw_command = float(
            action[2]
        )

        throttle_command = float(
            action[3]
        )

        # ====================================================
        # RC
        # ====================================================

        (
            roll_rc,
            pitch_rc,
            yaw_rc,
            throttle_rc,
        ) = self._action_to_rc(
            action
        )

        self._send_rc_override(
            throttle=throttle_rc,
            roll=roll_rc,
            pitch=pitch_rc,
            yaw=yaw_rc,
        )

        # ====================================================
        # DRONE STATE
        # ====================================================

        drone_position = (
            self._get_drone_position()
        )

        drone_velocity = (
            self._get_drone_velocity()
        )

        altitude = float(
            drone_position[2]
        )

        vertical_velocity = float(
            drone_velocity[2]
        )

        # ====================================================
        # FC / MOTOR
        # ====================================================

        motors = (
            self.fc.compute_motors(
                throttle_command=throttle_command,
                roll_command=roll_command,
                pitch_command=pitch_command,
                yaw_command=yaw_command,
                altitude=altitude,
                vertical_velocity=vertical_velocity,
                dt=CONTROL_DT,
            )
        )

        self._apply_motors(
            motors
        )

        # ====================================================
        # PHYSICS
        # ====================================================

        physics_dt = float(
            self.model.opt.timestep
        )

        substeps = max(
            1,
            int(
                round(
                    CONTROL_DT
                    / physics_dt
                )
            ),
        )

        sub_dt = (
            CONTROL_DT
            / substeps
        )

        for _ in range(
            substeps
        ):

            self._update_target(
                sub_dt
            )

            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.step_count += 1

        # ====================================================
        # OBSERVATION
        # ====================================================

        observation = (
            self._get_observation()
        )

        # ====================================================
        # REWARD
        # ====================================================

        reward = (
            self._calculate_reward()
        )

        self.episode_reward += (
            reward
        )

        # ====================================================
        # TERMINATION STATE
        # ====================================================

        position = (
            self._get_drone_position()
        )

        velocity = (
            self._get_drone_velocity()
        )

        altitude = float(
            position[2]
        )

        target = (
            self._get_target_position()
        )

        distance = float(
            np.linalg.norm(
                target - position
            )
        )

        self.episode_distance_sum += (
            distance
        )

        terminated = False
        truncated = False

        reason = ""

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(position)
        ):

            terminated = True
            reason = "physics"

        elif not np.all(
            np.isfinite(velocity)
        ):

            terminated = True
            reason = "physics"

        # ----------------------------------------------------
        # Ground
        # ----------------------------------------------------

        elif altitude < MIN_ALTITUDE:

            terminated = True
            reason = "ground"

        # ----------------------------------------------------
        # Altitude limit
        # ----------------------------------------------------

        elif altitude > MAX_ALTITUDE:

            terminated = True
            reason = "altitude_limit"

        # ----------------------------------------------------
        # Hard too close
        # ----------------------------------------------------

        elif distance < HARD_TOO_CLOSE_DISTANCE:

            terminated = True
            reason = "too_close"

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        elif self.episode_success:

            terminated = True
            reason = "success"

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        elif (
            self.step_count
            >= MAX_EPISODE_STEPS
        ):

            truncated = True
            reason = "timeout"

        self.last_reason = (
            reason
        )

        # ====================================================
        # VISUAL / ATTITUDE DIAGNOSTICS
        # ====================================================

        (
            visual_dx,
            visual_dy,
            _,
            _,
            visible,
        ) = (
            self._calculate_visual_detection()
        )

        roll, pitch, yaw = (
            self._get_euler()
        )

        # ====================================================
        # INFO
        # ====================================================

        info = {

            "altitude":
                altitude,

            "vertical_velocity":
                vertical_velocity,

            "distance":
                distance,

            "visible":
                bool(visible),

            "visual_dx":
                float(visual_dx),

            "visual_dy":
                float(visual_dy),

            "roll":
                float(roll),

            "pitch":
                float(pitch),

            "yaw":
                float(yaw),

            "rc_roll":
                roll_rc,

            "rc_pitch":
                pitch_rc,

            "rc_yaw":
                yaw_rc,

            "rc_throttle":
                throttle_rc,

            "throttle":
                self.fc.last_throttle,

            "vertical_acceleration":
                self.fc.last_vertical_acceleration,

            "motor1":
                float(motors[0]),

            "motor2":
                float(motors[1]),

            "motor3":
                float(motors[2]),

            "motor4":
                float(motors[3]),

            "reason":
                reason,

            "curriculum_level":
                self.curriculum_level,

            "episode_level":
                self.episode_level,

            "phase":
                self._get_episode_phase_name(),

            "success":
                bool(self.episode_success),

            "stable_steps":
                int(self.episode_stable_steps),

            "max_stable_steps":
                int(self.episode_max_stable_steps),

            "success_hold_steps":
                int(self.success_hold_steps),

            "reward_components":
                dict(self.reward_components),
        }

        # ====================================================
        # EPISODE STATISTICS
        # ====================================================

        if (
            terminated
            or truncated
        ):

            visible_ratio = (
                self.episode_visible_steps
                / max(
                    1,
                    self.step_count,
                )
            )

            follow_ratio = (
                self.episode_follow_steps
                / max(
                    1,
                    self.step_count,
                )
            )

            mean_distance = (
                self.episode_distance_sum
                / max(
                    1,
                    self.step_count,
                )
            )

            info[
                "episode_reward"
            ] = self.episode_reward

            info[
                "visible_ratio"
            ] = visible_ratio

            info[
                "follow_ratio"
            ] = follow_ratio

            info[
                "mean_distance"
            ] = mean_distance

            info[
                "episode_steps"
            ] = self.step_count

            info[
                "episode_level"
            ] = self.episode_level

            info[
                "episode_phase"
            ] = self._get_episode_phase_name()

            info[
                "max_stable_steps"
            ] = self.episode_max_stable_steps

            info[
                "success"
            ] = bool(
                self.episode_success
            )

            # ------------------------------------------------
            # Curriculum update happens ONLY HERE.
            #
            # Therefore curriculum transitions are between
            # episodes, never in the middle of an episode.
            # ------------------------------------------------

            self._update_curriculum(
                info
            )

            # ------------------------------------------------
            # Report the new curriculum level.
            # ------------------------------------------------

            info[
                "next_curriculum_level"
            ] = self.curriculum_level

            info[
                "next_phase"
            ] = self._get_phase_name()

            if self.diagnostic:

                print()
                print(
                    "[EPISODE] "
                    f"level={info['episode_level']} "
                    f"phase={info['episode_phase']} "
                    f"steps={info['episode_steps']} "
                    f"reward={info['episode_reward']:+.2f} "
                    f"visible={info['visible_ratio']:.3f} "
                    f"follow={info['follow_ratio']:.3f} "
                    f"mean_dist={info['mean_distance']:.2f} "
                    f"max_stable={info['max_stable_steps']} "
                    f"success={info['success']} "
                    f"reason={reason}"
                )

                if (
                    info["next_curriculum_level"]
                    != info["episode_level"]
                ):

                    print(
                        "[CURRICULUM] "
                        f"LEVEL "
                        f"{info['episode_level']} -> "
                        f"{info['next_curriculum_level']} "
                        f"("
                        f"{info['next_phase']}"
                        f")"
                    )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ========================================================
    # CURRICULUM
    # ========================================================

    def _update_curriculum(
        self,
        info=None,
    ):

        if not self.curriculum_enabled:
            return

        if self.step_count < 100:
            return

        visible_ratio = (
            self.episode_visible_steps
            / max(
                1,
                self.step_count,
            )
        )

        follow_ratio = (
            self.episode_follow_steps
            / max(
                1,
                self.step_count,
            )
        )

        level = int(
            self.curriculum_level
        )

        # ----------------------------------------------------
        # Level 0 -> 1
        #
        # Need altitude stability and enough episode duration.
        # ----------------------------------------------------

        if level == 0:

            if (
                self.step_count
                >= PHASE0_MIN_STEPS
                and
                self.episode_max_stable_steps
                >= 30
            ):

                self.curriculum_level = 1

                if self.diagnostic:

                    print(
                        "[CURRICULUM] "
                        "LEVEL 0 -> 1 "
                        "(visual acquisition)"
                    )

        # ----------------------------------------------------
        # Level 1 -> 2
        # ----------------------------------------------------

        elif level == 1:

            if (
                self.step_count
                >= PHASE1_MIN_STEPS
                and
                visible_ratio
                >= CURRICULUM_VISIBLE_THRESHOLD
            ):

                self.curriculum_level = 2

                if self.diagnostic:

                    print(
                        "[CURRICULUM] "
                        "LEVEL 1 -> 2 "
                        "(distance hold)"
                    )

        # ----------------------------------------------------
        # Level 2 -> 3
        # ----------------------------------------------------

        elif level == 2:

            if (
                self.step_count
                >= PHASE2_MIN_STEPS
                and
                visible_ratio
                >= CURRICULUM_VISIBLE_THRESHOLD
                and
                follow_ratio
                >= CURRICULUM_FOLLOW_THRESHOLD
            ):

                self.curriculum_level = 3

                if self.diagnostic:

                    print(
                        "[CURRICULUM] "
                        "LEVEL 2 -> 3 "
                        "(moving target)"
                    )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(
        self,
    ):

        if self.renderer is not None:

            try:

                self.renderer.close()

            except Exception:
                pass

            self.renderer = None

        if self.mavlink is not None:

            try:

                self.mavlink.close()

            except Exception:
                pass


# ============================================================
# ENV FACTORY
# ============================================================

def make_env(
    xml_path,
    diagnostic=False,
    curriculum=True,
):

    env = RealisticVisualFollowEnv(
        xml_path=xml_path,
        render_width=DEFAULT_WIDTH,
        render_height=DEFAULT_HEIGHT,
        diagnostic=diagnostic,
        curriculum=curriculum,
    )

    env = Monitor(
        env
    )

    return env


# ============================================================
# CURRICULUM HELPERS
# ============================================================

def unwrap_env(
    env,
):

    current = env

    while hasattr(
        current,
        "env",
    ):

        current = current.env

    return current


def get_curriculum_level(
    env,
):

    base_env = unwrap_env(
        env
    )

    return int(
        getattr(
            base_env,
            "curriculum_level",
            0,
        )
    )


def set_curriculum_level(
    env,
    level,
):

    base_env = unwrap_env(
        env
    )

    base_env.curriculum_level = int(
        clamp(
            int(level),
            0,
            CURRICULUM_MAX_LEVEL,
        )
    )


# ============================================================
# CURRICULUM-AWARE EVALUATION CALLBACK
# ============================================================

class CurriculumEvalCallback(
    EvalCallback
):

    def _on_step(
        self,
    ):

        # ----------------------------------------------------
        # Keep evaluation synchronized with the training
        # curriculum before EvalCallback performs evaluation.
        #
        # Training environment:
        #   level 0 -> 1 -> 2 -> 3
        #
        # Evaluation environment:
        #   same current level
        #
        # This prevents evaluating a level-3-capable agent
        # permanently against an unrelated fixed task.
        # ----------------------------------------------------

        try:

            train_level = (
                get_curriculum_level(
                    self.training_env
                )
            )

            set_curriculum_level(
                self.eval_env,
                train_level,
            )

        except Exception as e:

            if self.verbose > 1:

                print(
                    "[EVAL] "
                    f"Curriculum sync warning: {e}"
                )

        return super()._on_step()


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

def check_environment(
    xml_path,
):

    print()
    print("=" * 76)
    print(
        "CHECKING GYMNASIUM ENVIRONMENT V16.4"
    )
    print("=" * 76)

    env = RealisticVisualFollowEnv(
        xml_path=xml_path,
        diagnostic=True,
        curriculum=True,
    )

    try:

        check_env(
            env,
            warn=True,
            skip_render_check=True,
        )

        print()
        print(
            "[CHECK] Environment OK."
        )

        print(
            "[CHECK] Observation state: "
            "11"
        )

        print(
            "[CHECK] Privileged geometry: "
            "REMOVED"
        )

    finally:

        env.close()

    print("=" * 76)


# ============================================================
# PHYSICS DIAGNOSTIC
# ============================================================

def run_diagnostic(
    xml_path,
):

    print()
    print("=" * 76)
    print(
        "V16.4 PHYSICS / MOTOR DIAGNOSTIC"
    )
    print("=" * 76)

    env = RealisticVisualFollowEnv(
        xml_path=xml_path,
        diagnostic=True,
        curriculum=False,
    )

    try:

        print()
        print(
            f"[DIAG] mass="
            f"{env.drone_mass:.4f} kg"
        )

        print(
            f"[DIAG] weight="
            f"{env.weight:.4f} N"
        )

        print(
            f"[DIAG] hover force="
            f"{env.hover_force:.4f} N"
        )

        print(
            f"[DIAG] hover throttle="
            f"{env.hover_throttle:.6f}"
        )

        env.reset(
            seed=123
        )

        print()
        print(
            "[DIAG] Testing neutral hover..."
        )

        for step in range(
            200
        ):

            action = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )

            (
                _,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            if step % 10 == 0:

                print(
                    f"[DIAG] "
                    f"step={step:4d} "
                    f"alt={info['altitude']:7.3f} "
                    f"vz={info['vertical_velocity']:8.3f} "
                    f"reward={reward:+7.3f} "
                    f"thr={info['throttle']:.5f} "
                    f"motors=["
                    f"{info['motor1']:.2f},"
                    f"{info['motor2']:.2f},"
                    f"{info['motor3']:.2f},"
                    f"{info['motor4']:.2f}]"
                )

            if (
                terminated
                or truncated
            ):

                print(
                    "[DIAG] Episode ended:"
                    f" {info['reason']}"
                )

                break

        print()

        position = (
            env._get_drone_position()
        )

        velocity = (
            env._get_drone_velocity()
        )

        print(
            f"[DIAG] final alt="
            f"{position[2]:.3f} "
            f"vz={velocity[2]:.3f}"
        )

        print()
        print(
            "[DIAG] Physics diagnostic complete."
        )

    finally:

        env.close()


# ============================================================
# RANDOM TEST
# ============================================================

def run_random_test(
    xml_path,
    steps=400,
):

    print()
    print("=" * 76)
    print(
        "V16.4 RANDOM POLICY TEST"
    )
    print("=" * 76)

    env = RealisticVisualFollowEnv(
        xml_path=xml_path,
        diagnostic=True,
        curriculum=False,
    )

    try:

        obs, info = env.reset(
            seed=42
        )

        total_reward = 0.0

        for step in range(
            steps
        ):

            action = (
                env.action_space.sample()
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            total_reward += reward

            if step % 20 == 0:

                print(
                    f"[TEST] "
                    f"step={step:4d} "
                    f"reward={reward:+7.3f} "
                    f"total={total_reward:+9.3f} "
                    f"phase={info['phase']:18s} "
                    f"alt={info['altitude']:6.2f} "
                    f"dist={info['distance']:6.2f} "
                    f"visible={info['visible']}"
                )

            if (
                terminated
                or truncated
            ):

                print(
                    "[TEST] Episode ended: "
                    f"{info['reason']}"
                )

                break

        print()
        print(
            f"[TEST] Total reward: "
            f"{total_reward:.3f}"
        )

    finally:

        env.close()


# ============================================================
# TRAINING CALLBACK
# ============================================================

class TrainingInfoCallback(
    BaseCallback
):

    def __init__(
        self,
        verbose=0,
    ):

        super().__init__(
            verbose
        )

        self.last_print = 0

        self.last_level = None

    def _on_step(
        self,
    ):

        if (
            self.num_timesteps
            - self.last_print
            >= 5000
        ):

            self.last_print = (
                self.num_timesteps
            )

            if self.verbose:

                level = (
                    get_curriculum_level(
                        self.training_env
                    )
                )

                print(
                    f"[TRAIN] "
                    f"timesteps="
                    f"{self.num_timesteps} "
                    f"curriculum_level="
                    f"{level}"
                )

                self.last_level = level

        return True


# ============================================================
# TRAIN
# ============================================================

def train_agent(
    xml_path,
    timesteps,
    model_path,
    tensorboard_log,
):

    print()
    print("=" * 76)
    print(
        "STARTING SAC END-TO-END TRAINING V16.4"
    )
    print("=" * 76)

    os.makedirs(
        os.path.dirname(model_path)
        if os.path.dirname(model_path)
        else ".",
        exist_ok=True,
    )

    os.makedirs(
        tensorboard_log,
        exist_ok=True,
    )

    # ========================================================
    # TRAIN ENV
    # ========================================================

    print(
        "[TRAIN] Creating environment..."
    )

    train_env = make_env(
        xml_path=xml_path,
        diagnostic=False,
        curriculum=True,
    )

    # ========================================================
    # EVAL ENV
    # ========================================================

    print(
        "[TRAIN] Creating evaluation environment..."
    )

    # --------------------------------------------------------
    # Evaluation also has curriculum enabled.
    #
    # CurriculumEvalCallback synchronizes its level with
    # training before each evaluation.
    # --------------------------------------------------------

    eval_env = make_env(
        xml_path=xml_path,
        diagnostic=False,
        curriculum=True,
    )

    # ========================================================
    # SAC
    # ========================================================

    print(
        "[TRAIN] Creating SAC model..."
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            qf=[256, 256],
        )
    )

    model = SAC(
        policy="MultiInputPolicy",

        env=train_env,

        learning_rate=3e-4,

        buffer_size=100_000,

        learning_starts=5_000,

        batch_size=256,

        tau=0.005,

        gamma=0.99,

        train_freq=1,

        gradient_steps=1,

        ent_coef="auto",

        target_update_interval=1,

        policy_kwargs=policy_kwargs,

        tensorboard_log=tensorboard_log,

        verbose=1,

        device="auto",
    )

    # ========================================================
    # EVAL CALLBACK
    # ========================================================

    eval_callback = CurriculumEvalCallback(
        eval_env,

        best_model_save_path=(
            os.path.dirname(
                model_path
            )
            if os.path.dirname(
                model_path
            )
            else "."
        ),

        log_path=(
            os.path.dirname(
                model_path
            )
            if os.path.dirname(
                model_path
            )
            else "."
        ),

        eval_freq=10_000,

        n_eval_episodes=5,

        deterministic=True,

        render=False,

        verbose=1,
    )

    info_callback = (
        TrainingInfoCallback(
            verbose=1
        )
    )

    # ========================================================
    # TRAIN INFO
    # ========================================================

    print()
    print(
        f"[TRAIN] Timesteps: "
        f"{timesteps}"
    )

    print(
        f"[TRAIN] Device: "
        f"{model.device}"
    )

    print(
        "[TRAIN] Observation:"
    )

    print(
        "  RGB 48x48"
    )

    print(
        "  state(11)"
    )

    print(
        "  privileged target geometry: NONE"
    )

    print(
        "[TRAIN] Action:"
    )

    print(
        "  roll/pitch/yaw/throttle [-1,1]"
    )

    print(
        "[TRAIN] Curriculum:"
    )

    print(
        "  LEVEL 0 = stabilization"
    )

    print(
        "  LEVEL 1 = visual acquisition"
    )

    print(
        "  LEVEL 2 = distance hold"
    )

    print(
        "  LEVEL 3 = moving follow"
    )

    print()
    print(
        "[TRAIN] Curriculum transitions:"
    )

    print(
        "  only between completed episodes"
    )

    print(
        "  evaluation follows training level"
    )

    print()

    # ========================================================
    # TRAIN
    # ========================================================

    try:

        model.learn(
            total_timesteps=int(
                timesteps
            ),

            callback=[
                eval_callback,
                info_callback,
            ],

            progress_bar=True,

            log_interval=10,

            reset_num_timesteps=True,
        )

        # ====================================================
        # SAVE
        # ====================================================

        model.save(
            model_path
        )

        print()
        print(
            "[TRAIN] Model saved:"
        )

        print(
            f"[TRAIN] "
            f"{model_path}.zip"
        )

    except KeyboardInterrupt:

        print()
        print(
            "[TRAIN] Interrupted."
        )

        interrupted_path = (
            model_path
            + "_interrupted"
        )

        model.save(
            interrupted_path
        )

        print(
            "[TRAIN] Emergency model saved:"
        )

        print(
            f"[TRAIN] "
            f"{interrupted_path}.zip"
        )

    finally:

        train_env.close()

        eval_env.close()

    print()
    print("=" * 76)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 76)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path,
    env,
):

    path = model_path

    if path.endswith(
        ".zip"
    ):

        path = path[:-4]

    print(
        f"[MODEL] Loading: "
        f"{path}.zip"
    )

    model = SAC.load(
        path,
        env=env,
        device="auto",
    )

    return model


# ============================================================
# TEST TRAINED MODEL
# ============================================================

def test_agent(
    xml_path,
    model_path,
    episodes=5,
):

    print()
    print("=" * 76)
    print(
        "TESTING TRAINED SAC AGENT V16.4"
    )
    print("=" * 76)

    print(
        "[TEST] Mode: visual-only observation"
    )

    print(
        "[TEST] State dimension: 11"
    )

    print(
        "[TEST] Target geometry:"
    )

    print(
        "  distance/bearing/elevation "
        "NOT exposed to agent"
    )

    env = RealisticVisualFollowEnv(
        xml_path=xml_path,
        diagnostic=False,
        curriculum=False,
    )

    try:

        model = load_model(
            model_path,
            env,
        )

        rewards = []

        success_count = 0

        visible_ratios = []

        follow_ratios = []

        mean_distances = []

        max_stable_values = []

        for episode in range(
            episodes
        ):

            obs, info = env.reset(
                seed=1000 + episode
            )

            total_reward = 0.0

            visible_steps = 0

            follow_steps = 0

            max_stable_steps = 0

            print()

            print(
                f"[TEST] Episode "
                f"{episode + 1}/{episodes}"
            )

            for step in range(
                MAX_EPISODE_STEPS
            ):

                action, _ = (
                    model.predict(
                        obs,
                        deterministic=True,
                    )
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                total_reward += (
                    reward
                )

                if info.get(
                    "visible",
                    False,
                ):

                    visible_steps += 1

                distance = info.get(
                    "distance",
                    999.0,
                )

                if (
                    FOLLOW_MIN_DISTANCE
                    <= distance
                    <= FOLLOW_MAX_DISTANCE
                    and
                    info.get(
                        "visible",
                        False,
                    )
                ):

                    follow_steps += 1

                max_stable_steps = max(
                    max_stable_steps,
                    int(
                        info.get(
                            "stable_steps",
                            0,
                        )
                    ),
                )

                if step % 25 == 0:

                    print(
                        f"[TEST] "
                        f"step={step:4d} "
                        f"reward={reward:+7.3f} "
                        f"total={total_reward:+9.3f} "
                        f"alt={info['altitude']:6.2f} "
                        f"vz={info['vertical_velocity']:6.2f} "
                        f"dist={info['distance']:6.2f} "
                        f"visible={info['visible']} "
                        f"phase={info['phase']:18s} "
                        f"action=["
                        f"{action[0]:+.2f},"
                        f"{action[1]:+.2f},"
                        f"{action[2]:+.2f},"
                        f"{action[3]:+.2f}"
                        f"]"
                    )

                if (
                    terminated
                    or truncated
                ):

                    break

            episode_steps = (
                step + 1
            )

            rewards.append(
                total_reward
            )

            visible_ratio = (
                visible_steps
                / max(
                    1,
                    episode_steps,
                )
            )

            follow_ratio = (
                follow_steps
                / max(
                    1,
                    episode_steps,
                )
            )

            stable_ratio = (
                max_stable_steps
                / max(
                    1,
                    episode_steps,
                )
            )

            mean_distance = (
                info.get(
                    "mean_distance",
                    env.episode_distance_sum
                    / max(
                        1,
                        env.step_count,
                    ),
                )
            )

            success = bool(
                info.get(
                    "success",
                    False,
                )
            )

            if success:

                success_count += 1

            visible_ratios.append(
                visible_ratio
            )

            follow_ratios.append(
                follow_ratio
            )

            mean_distances.append(
                mean_distance
            )

            max_stable_values.append(
                max_stable_steps
            )

            print()

            print(
                f"[TEST] End reason: "
                f"{info['reason']}"
            )

            print(
                f"[TEST] Episode steps: "
                f"{episode_steps}"
            )

            print(
                f"[TEST] Reward: "
                f"{total_reward:.3f}"
            )

            print(
                f"[TEST] Visible ratio: "
                f"{visible_ratio:.3f}"
            )

            print(
                f"[TEST] Follow ratio: "
                f"{follow_ratio:.3f}"
            )

            print(
                f"[TEST] Mean distance: "
                f"{mean_distance:.3f}"
            )

            print(
                f"[TEST] Max stable steps: "
                f"{max_stable_steps}"
            )

            print(
                f"[TEST] Success: "
                f"{success}"
            )

        print()
        print("=" * 76)

        print(
            f"[TEST] Mean reward: "
            f"{np.mean(rewards):.3f}"
        )

        print(
            f"[TEST] Min reward: "
            f"{np.min(rewards):.3f}"
        )

        print(
            f"[TEST] Max reward: "
            f"{np.max(rewards):.3f}"
        )

        print(
            f"[TEST] Mean visible ratio: "
            f"{np.mean(visible_ratios):.3f}"
        )

        print(
            f"[TEST] Mean follow ratio: "
            f"{np.mean(follow_ratios):.3f}"
        )

        print(
            f"[TEST] Mean distance: "
            f"{np.mean(mean_distances):.3f}"
        )

        print(
            f"[TEST] Mean max stable steps: "
            f"{np.mean(max_stable_values):.1f}"
        )

        print(
            f"[TEST] Success: "
            f"{success_count}/{episodes}"
        )

        print("=" * 76)

    finally:

        env.close()


# ============================================================
# TRAIN + TEST
# ============================================================

def run_both(
    xml_path,
    timesteps,
    model_path,
    tensorboard_log,
):

    train_agent(
        xml_path=xml_path,
        timesteps=timesteps,
        model_path=model_path,
        tensorboard_log=tensorboard_log,
    )

    test_agent(
        xml_path=xml_path,
        model_path=model_path,
        episodes=5,
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "End-to-End Visual Follow RL V16.4"
        )
    )

    parser.add_argument(
        "--xml",
        default="scene.xml",
        help="MuJoCo XML scene",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "check",
            "diagnostic",
            "random",
            "train",
            "test",
            "both",
        ],
        default="check",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--model",
        default=(
            "visual_follow_v16_3"
        ),
    )

    parser.add_argument(
        "--tensorboard",
        default=(
            "./tensorboard/"
            "visual_follow_v16_3"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--test-steps",
        type=int,
        default=400,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print()
    print("=" * 76)
    print(
        "END-TO-END VISUAL FOLLOW V16.4"
    )
    print("=" * 76)

    print(
        f"[MAIN] XML: "
        f"{args.xml}"
    )

    print(
        f"[MAIN] Mode: "
        f"{args.mode}"
    )

    print(
        f"[MAIN] Timesteps: "
        f"{args.timesteps}"
    )

    print(
        f"[MAIN] Model: "
        f"{args.model}"
    )

    print(
        f"[MAIN] TensorBoard: "
        f"{args.tensorboard}"
    )

    print("=" * 76)

    # ========================================================
    # CHECK
    # ========================================================

    if args.mode == "check":

        check_environment(
            args.xml
        )

        return

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    if args.mode == "diagnostic":

        run_diagnostic(
            args.xml
        )

        return

    # ========================================================
    # RANDOM
    # ========================================================

    if args.mode == "random":

        run_random_test(
            args.xml,
            steps=args.test_steps,
        )

        return

    # ========================================================
    # TRAIN
    # ========================================================

    if args.mode == "train":

        train_agent(
            xml_path=args.xml,
            timesteps=args.timesteps,
            model_path=args.model,
            tensorboard_log=args.tensorboard,
        )

        return

    # ========================================================
    # TEST
    # ========================================================

    if args.mode == "test":

        test_agent(
            xml_path=args.xml,
            model_path=args.model,
            episodes=args.episodes,
        )

        return

    # ========================================================
    # BOTH
    # ========================================================

    if args.mode == "both":

        run_both(
            xml_path=args.xml,
            timesteps=args.timesteps,
            model_path=args.model,
            tensorboard_log=args.tensorboard,
        )

        return


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()