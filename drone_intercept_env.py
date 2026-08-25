import os
import math
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import mujoco
import gymnasium as gym

from gymnasium import spaces
from scipy.spatial.transform import Rotation as R


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("DroneInterceptEnv")


# ============================================================
# DETECTION
# ============================================================

@dataclass
class Detection:
    """
    ВАЖНО:
    НИКАКОЙ distance.

    Агент получает только то, что реально может получить
    из изображения:
        dx
        dy
        size
        visible
        confidence
    """

    dx: float = 0.0
    dy: float = 0.0
    size: float = 0.0
    visible: bool = False
    confidence: float = 0.0


@dataclass
class VisualObstacleState:
    """
    Визуальное представление препятствий.

    Это НЕ расстояние.

    Значения:
        forward_left
        forward_center
        forward_right
        left
        right

    Получаются из изображения камеры.
    """

    forward_left: float = 0.0
    forward_center: float = 0.0
    forward_right: float = 0.0
    left: float = 0.0
    right: float = 0.0

    # Насколько сильно изображение заполнено объектами.
    clutter: float = 0.0


# ============================================================
# SYNTHETIC YOLO
# ============================================================

@dataclass
class YOLOEmulatorConfig:

    confidence_threshold: float = 0.40

    base_false_negative_rate: float = 0.05

    small_target_miss_max: float = 0.35

    bbox_std_fraction: float = 0.04

    size_noise_std: float = 0.10

    confidence_noise_std: float = 0.03

    min_detectable_size: float = 0.02

    latency_seconds: float = 0.06

    dropout_enter_prob: float = 0.005

    dropout_stay_prob: float = 0.60


class TargetDetector(ABC):

    @abstractmethod
    def detect(self, frame, sim_time, step_dt):
        pass

    def reset(self):
        pass

    def seed(self, seed):
        pass


# ============================================================
# IMAGE TARGET DETECTOR
# ============================================================

class CameraTargetDetector(TargetDetector):
    """
    Визуальный target detector.

    В этой реализации предполагается, что цель имеет
    заметный цвет/контраст.

    НИКАКОЙ world position цели здесь нет.

    В реальности этот класс можно заменить на YOLO:
        YOLO(frame) -> bbox -> Detection
    """

    def __init__(
        self,
        target_hsv_low=(0, 70, 60),
        target_hsv_high=(20, 255, 255),
        confidence_threshold=0.25,
        image_width=96,
        image_height=96,
        noise_level="medium"
    ):

        self.target_hsv_low = np.array(
            target_hsv_low,
            dtype=np.uint8
        )

        self.target_hsv_high = np.array(
            target_hsv_high,
            dtype=np.uint8
        )

        self.confidence_threshold = confidence_threshold

        self.image_width = image_width
        self.image_height = image_height

        self.rng = np.random.default_rng()

        if noise_level == "low":
            self.noise = 0.015
            self.dropout = 0.02

        elif noise_level == "high":
            self.noise = 0.06
            self.dropout = 0.12

        else:
            self.noise = 0.035
            self.dropout = 0.06

        self.last_detection = Detection()

    def seed(self, seed):

        if seed is None:
            self.rng = np.random.default_rng()

        else:
            self.rng = np.random.default_rng(seed)

    def reset(self):

        self.last_detection = Detection()

    def detect(self, frame, sim_time, step_dt):

        if frame is None:
            return Detection()

        if frame.size == 0:
            return Detection()

        # RGB -> HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(
            hsv,
            self.target_hsv_low,
            self.target_hsv_high
        )

        # Убираем шум
        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:

            self.last_detection = Detection()

            return self.last_detection

        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(contour)

        image_area = frame.shape[0] * frame.shape[1]

        relative_area = area / max(image_area, 1)

        if relative_area < 0.001:

            self.last_detection = Detection()

            return self.last_detection

        x, y, w, h = cv2.boundingRect(contour)

        cx = x + w * 0.5
        cy = y + h * 0.5

        # Нормализованные координаты:
        # -1 = левый/верх
        # +1 = правый/низ

        dx = (cx / frame.shape[1] - 0.5) * 2.0

        dy = (cy / frame.shape[0] - 0.5) * 2.0

        size = math.sqrt(
            max(relative_area, 0.0)
        )

        size = float(
            np.clip(size * 4.0, 0.0, 1.0)
        )

        confidence = float(
            np.clip(
                0.4 + size * 0.6,
                0.0,
                1.0
            )
        )

        # False negative
        if self.rng.random() < self.dropout:

            self.last_detection = Detection(
                dx=0.0,
                dy=0.0,
                size=0.0,
                visible=False,
                confidence=0.0
            )

            return self.last_detection

        dx += self.rng.normal(0.0, self.noise)
        dy += self.rng.normal(0.0, self.noise)

        confidence += self.rng.normal(
            0.0,
            self.noise
        )

        detection = Detection(
            dx=float(np.clip(dx, -1.0, 1.0)),
            dy=float(np.clip(dy, -1.0, 1.0)),
            size=float(np.clip(size, 0.0, 1.0)),
            visible=True,
            confidence=float(np.clip(confidence, 0.0, 1.0))
        )

        self.last_detection = detection

        return detection


# ============================================================
# VIRTUAL FLIGHT CONTROLLER
# ============================================================

class VirtualFlightController:

    def __init__(
        self,
        drone_mass,
        max_thrust
    ):

        self.mass = float(drone_mass)

        self.max_thrust = float(max_thrust)

        self.hover_thrust = (
            self.mass * 9.81
        ) / 4.0

        self.angle_to_rate_p = 3.0

        self.rate_kp = 0.15
        self.rate_ki = 0.01
        self.rate_kd = 0.005

        self.integrals = np.zeros(
            3,
            dtype=np.float64
        )

        self.prev_errors = np.zeros(
            3,
            dtype=np.float64
        )

    def reset(self):

        self.integrals[:] = 0.0

        self.prev_errors[:] = 0.0

    def compute_motors(
        self,
        target_roll,
        target_pitch,
        target_yaw_rate,
        thrust_norm,
        mj_data,
        dt,
        drone_body_id
    ):

        target_roll = np.clip(
            target_roll,
            -0.6,
            0.6
        )

        target_pitch = np.clip(
            target_pitch,
            -0.6,
            0.6
        )

        target_yaw_rate = np.clip(
            target_yaw_rate,
            -1.5,
            1.5
        )

        thrust_norm = np.clip(
            thrust_norm,
            0.0,
            1.0
        )

        quat = mj_data.xquat[
            drone_body_id
        ]

        rot = R.from_quat(
            [
                quat[1],
                quat[2],
                quat[3],
                quat[0]
            ]
        )

        euler = rot.as_euler(
            "xyz"
        )

        roll_error = (
            target_roll -
            euler[0]
        )

        pitch_error = (
            target_pitch -
            euler[1]
        )

        desired_rates = np.array(
            [
                roll_error *
                self.angle_to_rate_p,

                pitch_error *
                self.angle_to_rate_p,

                target_yaw_rate
            ]
        )

        current_rates = mj_data.qvel[3:6]

        rate_errors = (
            desired_rates -
            current_rates
        )

        torques = np.zeros(
            3,
            dtype=np.float64
        )

        for i in range(3):

            self.integrals[i] += (
                rate_errors[i] * dt
            )

            self.integrals[i] = np.clip(
                self.integrals[i],
                -5.0,
                5.0
            )

            derivative = (
                rate_errors[i] -
                self.prev_errors[i]
            ) / dt if dt > 0 else 0.0

            torque = (
                self.rate_kp *
                rate_errors[i]
                +
                self.rate_ki *
                self.integrals[i]
                +
                self.rate_kd *
                derivative
            )

            torques[i] = np.clip(
                torque,
                -5.0,
                5.0
            )

            self.prev_errors[i] = (
                rate_errors[i]
            )

        roll_t, pitch_t, yaw_t = torques

        base_thrust = self.hover_thrust

        extra_thrust = (
            thrust_norm - 0.5
        ) * 2.0 * self.hover_thrust

        attitude_comp = max(
            math.cos(euler[0]) *
            math.cos(euler[1]),
            0.6
        )

        total_thrust = np.clip(
            (
                base_thrust +
                extra_thrust
            ) / attitude_comp,
            0.0,
            self.max_thrust
        )

        m1 = (
            total_thrust
            + roll_t
            + pitch_t
            - yaw_t
        )

        m2 = (
            total_thrust
            - roll_t
            + pitch_t
            + yaw_t
        )

        m3 = (
            total_thrust
            - roll_t
            - pitch_t
            - yaw_t
        )

        m4 = (
            total_thrust
            + roll_t
            - pitch_t
            + yaw_t
        )

        return np.clip(
            np.array(
                [m1, m2, m3, m4]
            ),
            0.0,
            self.max_thrust
        ).astype(np.float32)


# ============================================================
# PHASE
# ============================================================

class Phase(str, Enum):

    TAKEOFF = "takeoff"

    HOVER = "hover"

    SEARCH = "search"

    PURSUIT = "pursuit"

    INTERCEPT = "intercept"

    MISSION = "mission"


# ============================================================
# MISSION
# ============================================================

class MissionStage:

    TAKEOFF = 0

    TRANSIT_1 = 1

    TRANSIT_2 = 2

    SEARCH = 3

    PURSUIT = 4


# ============================================================
# ENVIRONMENT
# ============================================================

class DroneInterceptEnv(gym.Env):

    """
    REALISTIC CAMERA-ONLY DRONE ENVIRONMENT.

    Observation:

    [0-4]
        target dx
        target dy
        target visible
        target size
        target confidence

    [5-6]
        visual target velocity

    [7-10]
        roll
        pitch
        sin(yaw)
        cos(yaw)

    [11-12]
        altitude
        yaw rate

    [13-15]
        target memory
        lost time

    [16-17]
        lock
        lock counter

    [18-19]
        GPS x/y

    [20-25]
        visual obstacle information

    ВАЖНО:

    НЕТ:
        distance to target
        distance to obstacle
        raycast
        depth buffer
        collision proximity
        world obstacle coordinates
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30
    }

    STATE_SIZE = 26

    def __init__(
        self,
        xml_path="scene.xml",
        phase=Phase.MISSION,
        max_steps=2000,
        use_image=False,
        camera_resolution=(96, 96),
        action_repeat=4,
        detector=None,
        yolo_config=None
    ):

        super().__init__()

        if isinstance(
            phase,
            str
        ):
            phase = Phase(phase)

        self.phase = phase

        self.max_steps = int(
            max_steps
        )

        self.action_repeat = int(
            action_repeat
        )

        self.camera_resolution = (
            camera_resolution
        )

        self.model = (
            mujoco.MjModel.from_xml_path(
                xml_path
            )
        )

        self.data = mujoco.MjData(
            self.model
        )

        self.dt = float(
            self.model.opt.timestep
        )

        self.step_dt = (
            self.dt *
            self.action_repeat
        )

        self.drone_id = (
            self.model.body("x2").id
        )

        self.target_id = (
            self.model.body(
                "target_kral"
            ).id
        )

        self.target_mocap_id = (
            self.model.body_mocapid[
                self.target_id
            ]
        )

        drone_mass = float(
            self.model.body_mass[
                self.drone_id
            ]
        )

        self.fc = (
            VirtualFlightController(
                drone_mass=drone_mass,
                max_thrust=25.0
            )
        )

        # ====================================================
        # CAMERA
        # ====================================================

        self.renderer = mujoco.Renderer(
            self.model,
            height=camera_resolution[0],
            width=camera_resolution[1]
        )

        # ====================================================
        # TARGET DETECTOR
        # ====================================================

        self.detector = (
            detector
            if detector is not None
            else CameraTargetDetector(
                image_width=camera_resolution[1],
                image_height=camera_resolution[0]
            )
        )

        # ====================================================
        # TARGET
        # ====================================================

        self.target_speed = 0.0

        self.target_vel = np.zeros(
            3,
            dtype=np.float64
        )

        self.target_maneuver_timer = 0.0

        self.target_maneuver_interval = 1.5

        # ====================================================
        # ACTION
        # ====================================================

        self.action_space = spaces.Box(
            low=np.array(
                [
                    -0.6,
                    -0.6,
                    -1.5,
                    0.0
                ],
                dtype=np.float32
            ),
            high=np.array(
                [
                    0.6,
                    0.6,
                    1.5,
                    1.0
                ],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # ====================================================
        # OBSERVATION
        # ====================================================

        if use_image:

            self.use_image = True

            self.observation_space = spaces.Dict({

                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        camera_resolution[0],
                        camera_resolution[1],
                        3
                    ),
                    dtype=np.uint8
                ),

                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self.STATE_SIZE,
                    ),
                    dtype=np.float32
                )
            })

        else:

            self.use_image = False

            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.STATE_SIZE,
                ),
                dtype=np.float32
            )

        # ====================================================
        # RESET
        # ====================================================

        self._reset_episode_state()

    # ========================================================
    # RESET STATE
    # ========================================================

    def _reset_episode_state(self):

        self.current_step = 0

        self.sim_time = 0.0

        self.last_dx = 0.0
        self.last_dy = 0.0

        self.v_dx = 0.0
        self.v_dy = 0.0

        self.last_confidence = 0.0

        self.lost_time = 0.0

        self.lock_counter = 0

        self.locked = False

        self.mission_stage = (
            MissionStage.TAKEOFF
        )

        self.gate_1_passed = False
        self.gate_2_passed = False

        self._gate1_rewarded = False
        self._gate2_rewarded = False

        self.mission_target_ever_visible = False

        self._pursuit_lock_steps = 0

        self._takeoff_hold_steps = 0

        self._hover_hold_steps = 0

        self._last_detection = Detection()

        self.prev_frame = None

        self.search_timer = 0.0

        self.search_direction = 1.0

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.detector.seed(seed)

        self.detector.reset()

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        self._reset_episode_state()

        self.fc.reset()

        # ====================================================
        # DRONE START
        # ====================================================

        if self.phase == Phase.TAKEOFF:

            start_pos = [
                0.0,
                0.0,
                0.3
            ]

        elif self.phase == Phase.HOVER:

            start_pos = [
                0.0,
                0.0,
                15.0
            ]

        elif self.phase == Phase.SEARCH:

            start_pos = [
                0.0,
                0.0,
                8.0
            ]

        elif self.phase == Phase.PURSUIT:

            start_pos = [
                25.0,
                0.0,
                18.0
            ]

        elif self.phase == Phase.INTERCEPT:

            start_pos = [
                32.0,
                0.0,
                16.0
            ]

        else:

            start_pos = [
                0.0,
                0.0,
                0.3
            ]

        initial_yaw = self.np_random.uniform(
            -np.pi,
            np.pi
        )

        self.data.qpos[0:3] = (
            start_pos
        )

        q = R.from_euler(
            "z",
            initial_yaw
        ).as_quat()

        self.data.qpos[3:7] = [
            q[3],
            q[0],
            q[1],
            q[2]
        ]

        self.data.qvel[:] = 0.0

        # ====================================================
        # TARGET
        # ====================================================

        if self.phase in (
            Phase.TAKEOFF,
            Phase.HOVER
        ):

            target_pos = np.array(
                [
                    15.0,
                    0.0,
                    8.0
                ]
            )

            target_speed = 0.0

        elif self.phase == Phase.SEARCH:

            target_pos = np.array(
                [
                    38.0,
                    0.0,
                    16.0
                ]
            )

            target_speed = 0.0

        else:

            target_pos = np.array(
                [
                    self.np_random.uniform(
                        28.0,
                        42.0
                    ),

                    self.np_random.uniform(
                        -7.0,
                        7.0
                    ),

                    self.np_random.uniform(
                        11.0,
                        17.0
                    )
                ]
            )

            target_speed = self.np_random.uniform(
                3.0,
                5.0
            )

        self.data.mocap_pos[
            self.target_mocap_id
        ] = target_pos

        self.target_speed = (
            target_speed
        )

        heading = self.np_random.uniform(
            -np.pi,
            np.pi
        )

        self.target_vel = np.array(
            [
                target_speed *
                np.cos(heading),

                target_speed *
                np.sin(heading),

                0.0
            ]
        )

        mujoco.mj_forward(
            self.model,
            self.data
        )

        frame = self._render_camera()

        detection = (
            self.detector.detect(
                frame,
                self.sim_time,
                self.step_dt
            )
        )

        self._last_detection = (
            detection
        )

        obs = self._get_obs(
            detection,
            frame
        )

        return obs, {}

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        self.current_step += 1

        self.sim_time += (
            self.step_dt
        )

        action = np.clip(
            np.asarray(
                action,
                dtype=np.float32
            ),
            self.action_space.low,
            self.action_space.high
        )

        roll = float(action[0])
        pitch = float(action[1])
        yaw_rate = float(action[2])
        thrust = float(action[3])

        # ====================================================
        # PHYSICS
        # ====================================================

        for _ in range(
            self.action_repeat
        ):

            motors = (
                self.fc.compute_motors(
                    roll,
                    pitch,
                    yaw_rate,
                    thrust,
                    self.data,
                    self.dt,
                    self.drone_id
                )
            )

            self.data.ctrl[0:4] = motors

            self._update_target()

            mujoco.mj_step(
                self.model,
                self.data
            )

        # ====================================================
        # CAMERA
        # ====================================================

        frame = self._render_camera()

        # ====================================================
        # TARGET DETECTION
        # ====================================================

        detection = (
            self.detector.detect(
                frame,
                self.sim_time,
                self.step_dt
            )
        )

        self._last_detection = (
            detection
        )

        # ====================================================
        # VISUAL TARGET VELOCITY
        # ====================================================

        if detection.visible:

            self.v_dx = (
                detection.dx -
                self.last_dx
            ) / max(
                self.step_dt,
                1e-6
            )

            self.v_dy = (
                detection.dy -
                self.last_dy
            ) / max(
                self.step_dt,
                1e-6
            )

            self.last_dx = detection.dx
            self.last_dy = detection.dy

        else:

            self.v_dx = 0.0
            self.v_dy = 0.0

        if self.phase == Phase.MISSION:

            self._update_mission_stage(
                detection
            )

        obs = self._get_obs(
            detection,
            frame
        )

        reward, terminated, truncated, info = (
            self._dispatch_reward(
                detection
            )
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    # ========================================================
    # CAMERA
    # ========================================================

    def _render_camera(self):

        self.renderer.update_scene(
            self.data,
            camera="fpv_cam"
        )

        frame = np.asarray(
            self.renderer.render(),
            dtype=np.uint8
        )

        return frame

    # ========================================================
    # TARGET DETECTION
    # ========================================================

    def _detect_target_from_camera(
        self,
        frame
    ):

        return self.detector.detect(
            frame,
            self.sim_time,
            self.step_dt
        )

    # ========================================================
    # VISUAL OBSTACLE ESTIMATION
    # ========================================================

    def _get_visual_obstacles(
        self,
        frame,
        detection
    ):
        """
        НИКАКОГО mj_ray.

        НИКАКОГО depth.

        НИКАКОЙ геометрической дистанции.

        Используем только RGB изображения.

        Метод:
            grayscale
            edges
            local edge density

        Это не "расстояние".

        Это приблизительная информация:
            "перед камерой появилось много контуров".

        В реальном проекте здесь можно поставить:
            segmentation network
            optical flow network
            DINO/YOLO segmentation
            RAFT
            lightweight obstacle detector
        """

        if frame is None:
            return VisualObstacleState()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            gray,
            50,
            120
        )

        h, w = edges.shape

        # ====================================================
        # Убираем область цели
        # ====================================================

        if detection.visible:

            cx = int(
                (detection.dx + 1.0)
                * 0.5 * w
            )

            cy = int(
                (detection.dy + 1.0)
                * 0.5 * h
            )

            radius = max(
                int(
                    detection.size *
                    min(w, h) *
                    0.5
                ),
                4
            )

            cv2.circle(
                edges,
                (
                    cx,
                    cy
                ),
                radius,
                0,
                -1
            )

        # ====================================================
        # SECTORS
        # ====================================================

        def density(
            x1,
            y1,
            x2,
            y2
        ):

            roi = edges[
                y1:y2,
                x1:x2
            ]

            if roi.size == 0:
                return 0.0

            return float(
                np.mean(
                    roi > 0
                )
            )

        third = w // 3

        left = density(
            0,
            0,
            third,
            h
        )

        center = density(
            third,
            0,
            third * 2,
            h
        )

        right = density(
            third * 2,
            0,
            w,
            h
        )

        # Центральная часть выше интересует
        # для предотвращения лобового столкновения.

        upper_center = density(
            third,
            0,
            third * 2,
            h // 2
        )

        center_left = density(
            0,
            h // 4,
            w // 2,
            3 * h // 4
        )

        center_right = density(
            w // 2,
            h // 4,
            w,
            3 * h // 4
        )

        clutter = float(
            np.mean(edges > 0)
        )

        # Нормируем примерно в [0,1]

        return VisualObstacleState(

            forward_left=float(
                np.clip(
                    left * 5.0,
                    0.0,
                    1.0
                )
            ),

            forward_center=float(
                np.clip(
                    (
                        center +
                        upper_center
                    ) * 2.5,
                    0.0,
                    1.0
                )
            ),

            forward_right=float(
                np.clip(
                    right * 5.0,
                    0.0,
                    1.0
                )
            ),

            left=float(
                np.clip(
                    center_left * 5.0,
                    0.0,
                    1.0
                )
            ),

            right=float(
                np.clip(
                    center_right * 5.0,
                    0.0,
                    1.0
                )
            ),

            clutter=float(
                np.clip(
                    clutter * 5.0,
                    0.0,
                    1.0
                )
            )
        )

    # ========================================================
    # MISSION
    # ========================================================

    def _update_mission_stage(
        self,
        detection
    ):

        altitude = float(
            self.data.xpos[
                self.drone_id
            ][2]
        )

        drone_x = float(
            self.data.xpos[
                self.drone_id
            ][0]
        )

        if (
            self.mission_stage ==
            MissionStage.TAKEOFF
            and altitude > 4.0
        ):

            self.mission_stage = (
                MissionStage.TRANSIT_1
            )

        elif (
            self.mission_stage ==
            MissionStage.TRANSIT_1
            and drone_x > 10.0
        ):

            self.mission_stage = (
                MissionStage.TRANSIT_2
            )

            self.gate_1_passed = True

        elif (
            self.mission_stage ==
            MissionStage.TRANSIT_2
            and drone_x > 22.0
        ):

            self.mission_stage = (
                MissionStage.SEARCH
            )

            self.gate_2_passed = True

        elif (
            self.mission_stage ==
            MissionStage.SEARCH
            and detection.visible
        ):

            self.mission_stage = (
                MissionStage.PURSUIT
            )

            self.mission_target_ever_visible = True

    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_obs(
        self,
        detection,
        frame
    ):

        if detection.visible:

            self.last_confidence = (
                detection.confidence
            )

            self.lost_time = 0.0

        else:

            self.lost_time += (
                self.step_dt
            )

        # ====================================================
        # LOCK
        # ====================================================

        if (
            detection.visible
            and
            abs(detection.dx) < 0.15
            and
            abs(detection.dy) < 0.15
            and
            detection.size > 0.01
            and
            detection.confidence > 0.5
        ):

            self.lock_counter += 1

        else:

            self.lock_counter = 0

        self.locked = (
            self.lock_counter >= 8
        )

        # ====================================================
        # IMU
        # ====================================================

        quat = self.data.xquat[
            self.drone_id
        ]

        rot = R.from_quat(
            [
                quat[1],
                quat[2],
                quat[3],
                quat[0]
            ]
        )

        euler = rot.as_euler(
            "xyz"
        )

        roll = float(
            np.clip(
                euler[0] / 1.5,
                -1.0,
                1.0
            )
        )

        pitch = float(
            np.clip(
                euler[1] / 1.5,
                -1.0,
                1.0
            )
        )

        yaw = euler[2]

        yaw_rate = float(
            self.data.qvel[5]
        )

        altitude = float(
            self.data.xpos[
                self.drone_id
            ][2]
        )

        # ====================================================
        # GPS
        # ====================================================

        drone_pos = self.data.xpos[
            self.drone_id
        ]

        gps_x = float(
            np.clip(
                drone_pos[0] / 45.0,
                -1.0,
                1.0
            )
        )

        gps_y = float(
            np.clip(
                drone_pos[1] / 10.0,
                -1.0,
                1.0
            )
        )

        # ====================================================
        # VISUAL OBSTACLES
        # ====================================================

        obstacles = (
            self._get_visual_obstacles(
                frame,
                detection
            )
        )

        # ====================================================
        # STATE
        # ====================================================

        state = np.array(

            [

                # -------------------------------
                # TARGET
                # -------------------------------

                detection.dx,

                detection.dy,

                float(
                    detection.visible
                ),

                detection.size,

                detection.confidence,

                # -------------------------------
                # TARGET VISUAL VELOCITY
                # -------------------------------

                float(
                    np.clip(
                        self.v_dx / 10.0,
                        -1.0,
                        1.0
                    )
                ),

                float(
                    np.clip(
                        self.v_dy / 10.0,
                        -1.0,
                        1.0
                    )
                ),

                # -------------------------------
                # IMU
                # -------------------------------

                roll,

                pitch,

                np.sin(yaw),

                np.cos(yaw),

                # -------------------------------
                # FC TELEMETRY
                # -------------------------------

                float(
                    np.clip(
                        altitude / 20.0,
                        0.0,
                        1.5
                    )
                ),

                float(
                    np.clip(
                        yaw_rate / 1.5,
                        -1.0,
                        1.0
                    )
                ),

                # -------------------------------
                # MEMORY
                # -------------------------------

                self.last_dx,

                self.last_dy,

                float(
                    np.clip(
                        self.lost_time / 3.0,
                        0.0,
                        1.0
                    )
                ),

                # -------------------------------
                # LOCK
                # -------------------------------

                float(
                    self.locked
                ),

                float(
                    np.clip(
                        self.lock_counter / 10.0,
                        0.0,
                        1.0
                    )
                ),

                # -------------------------------
                # GPS
                # -------------------------------

                gps_x,

                gps_y,

                # -------------------------------
                # CAMERA OBSTACLE FEATURES
                # -------------------------------

                obstacles.forward_left,

                obstacles.forward_center,

                obstacles.forward_right,

                obstacles.left,

                obstacles.right,

                obstacles.clutter,

            ],

            dtype=np.float32
        )

        if self.use_image:

            return {
                "image": frame,
                "state": state
            }

        return state

    # ========================================================
    # TARGET MOTION
    # ========================================================

    def _update_target(self):

        if self.phase not in (
            Phase.PURSUIT,
            Phase.INTERCEPT,
            Phase.MISSION
        ):

            return

        self.target_maneuver_timer += (
            self.dt
        )

        if (
            self.target_maneuver_timer
            >=
            self.target_maneuver_interval
        ):

            self.target_maneuver_timer = 0.0

            current_speed = float(
                np.linalg.norm(
                    self.target_vel[:2]
                )
            )

            current_heading = float(
                np.arctan2(
                    self.target_vel[1],
                    self.target_vel[0]
                )
            )

            new_heading = (
                current_heading
                +
                self.np_random.uniform(
                    -np.pi / 2,
                    np.pi / 2
                )
            )

            vertical_velocity = (
                self.np_random.uniform(
                    -0.6,
                    0.6
                )
            )

            self.target_vel = np.array(
                [
                    current_speed *
                    np.cos(new_heading),

                    current_speed *
                    np.sin(new_heading),

                    vertical_velocity
                ]
            )

        self.data.mocap_pos[
            self.target_mocap_id
        ] += (
            self.target_vel *
            self.dt
        )

        pos = self.data.mocap_pos[
            self.target_mocap_id
        ]

        if pos[0] <= 25.0:

            pos[0] = 25.0

            self.target_vel[0] = abs(
                self.target_vel[0]
            )

        elif pos[0] >= 45.0:

            pos[0] = 45.0

            self.target_vel[0] = -abs(
                self.target_vel[0]
            )

        if pos[1] <= -8.0:

            pos[1] = -8.0

            self.target_vel[1] = abs(
                self.target_vel[1]
            )

        elif pos[1] >= 8.0:

            pos[1] = 8.0

            self.target_vel[1] = -abs(
                self.target_vel[1]
            )

        if pos[2] <= 10.0:

            pos[2] = 10.0

            self.target_vel[2] = abs(
                self.target_vel[2]
            )

        elif pos[2] >= 18.0:

            pos[2] = 18.0

            self.target_vel[2] = -abs(
                self.target_vel[2]
            )

    # ========================================================
    # COLLISION
    # ========================================================

    def _check_collision(self):

        for i in range(
            self.data.ncon
        ):

            contact = (
                self.data.contact[i]
            )

            g1 = self.model.geom(
                contact.geom1
            ).name

            g2 = self.model.geom(
                contact.geom2
            ).name

            if (
                g1 == "floor"
                or
                g2 == "floor"
            ):
                continue

            b1 = (
                self.model.geom_bodyid[
                    contact.geom1
                ]
            )

            b2 = (
                self.model.geom_bodyid[
                    contact.geom2
                ]
            )

            if (
                b1 == self.drone_id
                or
                b2 == self.drone_id
            ):

                return True

        return False

    # ========================================================
    # REWARD DISPATCH
    # ========================================================

    def _dispatch_reward(
        self,
        detection
    ):

        dispatch = {

            Phase.TAKEOFF:
                self._reward_takeoff,

            Phase.HOVER:
                self._reward_hover,

            Phase.SEARCH:
                self._reward_search,

            Phase.PURSUIT:
                self._reward_pursuit,

            Phase.INTERCEPT:
                self._reward_intercept,

            Phase.MISSION:
                self._reward_mission
        }

        return dispatch[
            self.phase
        ](detection)

    # ========================================================
    # TAKEOFF
    # ========================================================

    def _reward_takeoff(
        self,
        detection
    ):

        altitude = float(
            self.data.xpos[
                self.drone_id
            ][2]
        )

        euler = self._get_euler()

        tilt = (
            euler[0] ** 2
            +
            euler[1] ** 2
        )

        z_vel = float(
            self.data.qvel[2]
        )

        reward = 0.0

        reward += (
            2.0 *
            np.exp(
                -0.5 *
                (altitude - 5.0) ** 2
            )
        )

        reward -= (
            0.3 *
            min(
                abs(z_vel),
                2.0
            )
        )

        reward -= (
            0.3 *
            tilt
        )

        reward -= 0.005

        if (
            altitude >= 4.5
            and
            tilt < 0.2
        ):

            self._takeoff_hold_steps += 1

        else:

            self._takeoff_hold_steps = 0

        if (
            self._takeoff_hold_steps
            >= 50
        ):

            return (
                reward + 100.0,
                True,
                False,
                {
                    "is_success": True,
                    "phase": "takeoff"
                }
            )

        if self.current_step > 400:

            if (
                abs(euler[0]) > 1.4
                or
                abs(euler[1]) > 1.4
            ):

                return (
                    reward - 20.0,
                    True,
                    False,
                    {
                        "is_success": False,
                        "reason":
                            "crash_attitude"
                    }
                )

            if self._check_collision():

                return (
                    reward - 20.0,
                    True,
                    False,
                    {
                        "is_success": False,
                        "reason":
                            "crash_obstacle"
                    }
                )

        return (
            reward,
            False,
            self.current_step >= self.max_steps,
            {
                "altitude":
                    altitude
            }
        )

    # ========================================================
    # HOVER
    # ========================================================

    def _reward_hover(
        self,
        detection
    ):

        altitude = float(
            self.data.xpos[
                self.drone_id
            ][2]
        )

        euler = self._get_euler()

        z_vel = float(
            self.data.qvel[2]
        )

        alt_error = (
            altitude - 15.0
        )

        tilt = (
            euler[0] ** 2
            +
            euler[1] ** 2
        )

        reward = (
            3.0 *
            np.exp(
                -0.05 *
                alt_error ** 2
            )
        )

        reward -= (
            0.3 *
            min(
                abs(z_vel),
                2.0
            )
        )

        reward -= (
            0.3 *
            tilt
        )

        reward -= 0.005

        if (
            abs(alt_error) < 0.5
            and
            tilt < 0.2
        ):

            self._hover_hold_steps += 1

        else:

            self._hover_hold_steps = 0

        if (
            self._hover_hold_steps
            >= 300
        ):

            return (
                reward + 30.0,
                True,
                False,
                {
                    "is_success": True,
                    "phase": "hover"
                }
            )

        return (
            reward,
            False,
            self.current_step >= self.max_steps,
            {
                "altitude":
                    altitude
            }
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def _reward_search(
        self,
        detection
    ):

        reward = -0.01

        if detection.visible:

            reward += 10.0

            center_error = (
                np.sqrt(
                    detection.dx ** 2
                    +
                    detection.dy ** 2
                )
            )

            reward += (
                1.0 *
                (
                    1.0 -
                    np.clip(
                        center_error,
                        0.0,
                        1.4
                    )
                )
            )

            reward += (
                detection.confidence
            )

            if self.current_step > 50:

                return (
                    reward,
                    True,
                    False,
                    {
                        "is_success": True,
                        "phase": "search",
                        "visible": True
                    }
                )

        else:

            yaw_rate = float(
                self.data.qvel[5]
            )

            if abs(yaw_rate) > 0.4:

                reward += 0.2

        return (
            reward,
            False,
            self.current_step >= self.max_steps,
            {
                "visible":
                    detection.visible
            }
        )

    # ========================================================
    # PURSUIT
    # ========================================================

    def _reward_pursuit(
        self,
        detection
    ):

        reward = -0.01

        if detection.visible:

            center_error = (
                np.sqrt(
                    detection.dx ** 2
                    +
                    detection.dy ** 2
                )
            )

            # Основной reward:
            # держать цель в центре.
            reward += (
                2.0 *
                (
                    1.0 -
                    np.clip(
                        center_error,
                        0.0,
                        1.0
                    )
                )
            )

            # Размер цели = визуальный cue.
            reward += (
                1.5 *
                detection.size
            )

            reward += (
                0.5 *
                detection.confidence
            )

            # Слишком резкое движение
            # цели по кадру нежелательно.
            visual_speed = (
                abs(self.v_dx)
                +
                abs(self.v_dy)
            )

            reward -= (
                0.5 *
                np.clip(
                    visual_speed / 5.0,
                    0.0,
                    1.0
                )
            )

        else:

            reward -= 0.5

        if detection.visible:

            if (
                abs(detection.dx) < 0.12
                and
                abs(detection.dy) < 0.12
                and
                detection.size > 0.03
            ):

                self._pursuit_lock_steps += 1

            else:

                self._pursuit_lock_steps = 0

        else:

            self._pursuit_lock_steps = 0

        if (
            self._pursuit_lock_steps
            >= 30
        ):

            return (
                reward + 50.0,
                True,
                False,
                {
                    "is_success": True,
                    "phase": "pursuit"
                }
            )

        if self.current_step > 400:

            if self._check_collision():

                return (
                    reward - 50.0,
                    True,
                    False,
                    {
                        "is_success": False,
                        "reason":
                            "crash_obstacle"
                    }
                )

        return (
            reward,
            False,
            self.current_step >= self.max_steps,
            {
                "visible":
                    detection.visible
            }
        )

    # ========================================================
    # INTERCEPT
    # ========================================================

    def _reward_intercept(
        self,
        detection
    ):

        reward = -0.01

        if detection.visible:

            center_error = (
                np.sqrt(
                    detection.dx ** 2
                    +
                    detection.dy ** 2
                )
            )

            reward += (
                3.0 *
                (
                    1.0 -
                    np.clip(
                        center_error,
                        0.0,
                        1.0
                    )
                )
            )

            reward += (
                2.0 *
                detection.size
            )

            reward += (
                0.5 *
                detection.confidence
            )

        else:

            reward -= 0.7

        # ====================================================
        # ВАЖНО:
        #
        # Успех НЕ определяется distance.
        #
        # Используется только визуальный lock.
        # ====================================================

        if (
            detection.visible
            and
            abs(detection.dx) < 0.10
            and
            abs(detection.dy) < 0.10
            and
            detection.size > 0.15
            and
            detection.confidence > 0.5
        ):

            self._pursuit_lock_steps += 1

        else:

            self._pursuit_lock_steps = 0

        if (
            self._pursuit_lock_steps
            >= 20
        ):

            return (
                reward + 200.0,
                True,
                False,
                {
                    "is_success": True,
                    "phase": "intercept"
                }
            )

        if self.current_step > 400:

            if self._check_collision():

                return (
                    reward - 100.0,
                    True,
                    False,
                    {
                        "is_success": False,
                        "reason":
                            "crash_obstacle"
                    }
                )

        return (
            reward,
            False,
            self.current_step >= self.max_steps,
            {
                "visible":
                    detection.visible
            }
        )

    # ========================================================
    # MISSION
    # ========================================================

    def _reward_mission(
        self,
        detection
    ):

        reward = -0.005

        drone_pos = self.data.xpos[
            self.drone_id
        ]

        euler = self._get_euler()

        tilt = (
            euler[0] ** 2
            +
            euler[1] ** 2
        )

        # ====================================================
        # TAKEOFF
        # ====================================================

        if (
            self.mission_stage
            ==
            MissionStage.TAKEOFF
        ):

            altitude = float(
                drone_pos[2]
            )

            reward += (
                2.0 *
                np.exp(
                    -0.5 *
                    (altitude - 5.0)
                    ** 2
                )
            )

        # ====================================================
        # TRANSIT
        # ====================================================

        elif (
            self.mission_stage
            ==
            MissionStage.TRANSIT_1
        ):

            progress = float(
                np.clip(
                    drone_pos[0]
                    / 10.0,
                    0.0,
                    1.0
                )
            )

            reward += (
                2.0 *
                progress
            )

            if (
                drone_pos[0] > 10.5
                and
                not self._gate1_rewarded
            ):

                reward += 20.0

                self._gate1_rewarded = True

        elif (
            self.mission_stage
            ==
            MissionStage.TRANSIT_2
        ):

            progress = float(
                np.clip(
                    (
                        drone_pos[0] -
                        10.0
                    ) / 12.0,
                    0.0,
                    1.0
                )
            )

            reward += (
                2.0 *
                progress
            )

            if (
                drone_pos[0] > 22.5
                and
                not self._gate2_rewarded
            ):

                reward += 20.0

                self._gate2_rewarded = True

        # ====================================================
        # SEARCH
        # ====================================================

        elif (
            self.mission_stage
            ==
            MissionStage.SEARCH
        ):

            if detection.visible:

                reward += 12.0

                self.mission_target_ever_visible = True

            else:

                yaw_rate = float(
                    self.data.qvel[5]
                )

                if abs(yaw_rate) > 0.4:

                    reward += 0.25

        # ====================================================
        # PURSUIT
        # ====================================================

        elif (
            self.mission_stage
            ==
            MissionStage.PURSUIT
        ):

            if detection.visible:

                center_error = (
                    np.sqrt(
                        detection.dx ** 2
                        +
                        detection.dy ** 2
                    )
                )

                reward += (
                    3.0 *
                    (
                        1.0 -
                        np.clip(
                            center_error,
                            0.0,
                            1.0
                        )
                    )
                )

                reward += (
                    2.0 *
                    detection.size
                )

                reward += (
                    0.5 *
                    detection.confidence
                )

                visual_speed = (
                    abs(self.v_dx)
                    +
                    abs(self.v_dy)
                )

                reward -= (
                    0.5 *
                    np.clip(
                        visual_speed / 5.0,
                        0.0,
                        1.0
                    )
                )

            else:

                reward -= 0.5

        # ====================================================
        # CAMERA OBSTACLE SIGNAL
        # ====================================================

        frame = self._render_camera()

        obstacle = (
            self._get_visual_obstacles(
                frame,
                detection
            )
        )

        # ====================================================
        # CAMERA-ONLY AVOIDANCE
        # ====================================================

        if obstacle.forward_center > 0.35:

            reward -= (
                2.0 *
                obstacle.forward_center
            )

        if obstacle.forward_left > 0.45:

            reward -= (
                0.8 *
                obstacle.forward_left
            )

        if obstacle.forward_right > 0.45:

            reward -= (
                0.8 *
                obstacle.forward_right
            )

        # ====================================================
        # ATTITUDE
        # ====================================================

        reward -= (
            0.2 *
            tilt
        )

        if (
            abs(euler[0]) > 0.9
            or
            abs(euler[1]) > 0.9
        ):

            reward -= 0.5

        # ====================================================
        # SUCCESS:
        #
        # ТОЛЬКО визуальный захват.
        # ====================================================

        if (
            detection.visible
            and
            abs(detection.dx) < 0.10
            and
            abs(detection.dy) < 0.10
            and
            detection.size > 0.15
            and
            detection.confidence > 0.5
        ):

            self._pursuit_lock_steps += 1

        else:

            self._pursuit_lock_steps = 0

        if (
            self._pursuit_lock_steps
            >= 20
        ):

            return (
                reward + 200.0,
                True,
                False,
                {
                    "is_success": True,
                    "phase": "mission",
                    "stage":
                        self.mission_stage
                }
            )

        # ====================================================
        # CRASH
        #
        # Collision НЕ подаётся агенту как информация.
        # ====================================================

        if self.current_step > 400:

            if (
                abs(euler[0]) > 1.3
                or
                abs(euler[1]) > 1.3
            ):

                return (
                    reward - 50.0,
                    True,
                    False,
                    {
                        "is_success": False,
                        "reason":
                            "crash_attitude"
                    }
                )

            if self._check_collision():

                return (
                    reward - 100.0,
                    True,
                    False,
                    {
                        "is_success": False,
                        "reason":
                            "crash_obstacle"
                    }
                )

            altitude = float(
                drone_pos[2]
            )

            if (
                altitude < 0.5
                and
                self.mission_stage
                >
                MissionStage.TAKEOFF
            ):

                return (
                    reward - 50.0,
                    True,
                    False,
                    {
                        "is_success": False,
                        "reason":
                            "ground_crash"
                    }
                )

        return (
            reward,
            False,
            self.current_step >= self.max_steps,
            {
                "stage":
                    self.mission_stage,
                "visible":
                    detection.visible
            }
        )

    # ========================================================
    # EULER
    # ========================================================

    def _get_euler(self):

        quat = self.data.xquat[
            self.drone_id
        ]

        rot = R.from_quat(
            [
                quat[1],
                quat[2],
                quat[3],
                quat[0]
            ]
        )

        return rot.as_euler(
            "xyz"
        )

    # ========================================================
    # RENDER
    # ========================================================

    def render(self):

        return self._render_camera()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.renderer is not None:

            self.renderer.close()