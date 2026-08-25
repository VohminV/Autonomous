import os
import math
import struct
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("DroneInterceptEnv")

@dataclass
class Detection:
    dx: float = 0.0
    dy: float = 0.0
    size: float = 0.0
    visible: bool = False
    confidence: float = 0.0
    distance: float = float("inf")

@dataclass
class YOLOEmulatorConfig:
    confidence_threshold: float = 0.40
    base_false_negative_rate: float = 0.05
    small_target_miss_max: float = 0.35
    bbox_std_fraction: float = 0.04
    size_noise_std: float = 0.10
    confidence_noise_std: float = 0.03
    min_detectable_size: float = 0.02
    max_detection_distance: float = 50.0
    latency_seconds: float = 0.06
    dropout_enter_prob: float = 0.005
    dropout_stay_prob: float = 0.60

class YOLOEmulator:
    def __init__(self, cfg=None, rng=None):
        self.cfg = cfg or YOLOEmulatorConfig()
        self.rng = rng or np.random.default_rng()
        self._latency_buffer = deque()
        self._dropout_active = False

    def reset(self):
        self._latency_buffer.clear()
        self._dropout_active = False

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def detect(self, oracle, sim_time, step_dt):
        noisy = self._instantaneous(oracle)
        if self.cfg.latency_seconds <= 0.0:
            return noisy
        self._latency_buffer.append((sim_time, noisy))
        horizon = self.cfg.latency_seconds + 2.0 * step_dt
        while self._latency_buffer and self._latency_buffer[0][0] < sim_time - horizon:
            self._latency_buffer.popleft()
        if not self._latency_buffer:
            return noisy
        target_time = sim_time - self.cfg.latency_seconds
        best_time, best_det = self._latency_buffer[0]
        for t, det in self._latency_buffer:
            if abs(t - target_time) < abs(best_time - target_time):
                best_time, best_det = t, det
        return best_det

    def _instantaneous(self, oracle):
        distance = oracle.distance
        if not oracle.visible or distance > self.cfg.max_detection_distance or oracle.size < self.cfg.min_detectable_size:
            return Detection(distance=distance)
        confidence = self._estimate_confidence(oracle.size, distance)
        if confidence < self.cfg.confidence_threshold:
            return Detection(distance=distance, confidence=confidence)
        self._update_dropout_state()
        if self._dropout_active or self.rng.random() < self._miss_probability(oracle.size):
            return Detection(distance=distance, confidence=0.0)
        noise = self.cfg.bbox_std_fraction
        dx = float(np.clip(oracle.dx + self.rng.normal(0.0, noise), -1.0, 1.0))
        dy = float(np.clip(oracle.dy + self.rng.normal(0.0, noise), -1.0, 1.0))
        size = float(np.clip(oracle.size * (1.0 + self.rng.normal(0.0, self.cfg.size_noise_std)), 0.0, 1.0))
        confidence = float(np.clip(confidence + self.rng.normal(0.0, self.cfg.confidence_noise_std), 0.0, 1.0))
        return Detection(dx=dx, dy=dy, size=size, visible=True, confidence=confidence, distance=distance)

    def _update_dropout_state(self):
        if self._dropout_active:
            if self.rng.random() > self.cfg.dropout_stay_prob:
                self._dropout_active = False
        else:
            if self.rng.random() < self.cfg.dropout_enter_prob:
                self._dropout_active = True

    def _miss_probability(self, size):
        if size >= 0.3:
            return self.cfg.base_false_negative_rate
        smallness = (0.3 - size) / 0.3
        return self.cfg.base_false_negative_rate + smallness * (self.cfg.small_target_miss_max - self.cfg.base_false_negative_rate)

    def _estimate_confidence(self, size, distance):
        size_term = float(np.clip(size / 0.4, 0.0, 1.0))
        dist_term = float(np.clip(1.0 - distance / self.cfg.max_detection_distance, 0.0, 1.0))
        return float(np.clip(0.5 + 0.3 * size_term + 0.2 * dist_term, 0.0, 1.0))

class TargetDetector(ABC):
    @abstractmethod
    def detect(self, oracle, sim_time, step_dt): ...
    def reset(self): pass
    def seed(self, seed): pass

class OracleDetector(TargetDetector):
    def detect(self, oracle, sim_time, step_dt):
        return oracle

class SyntheticYOLODetector(TargetDetector):
    def __init__(self, cfg=None, rng=None):
        self.emulator = YOLOEmulator(cfg=cfg, rng=rng)
    def detect(self, oracle, sim_time, step_dt):
        return self.emulator.detect(oracle, sim_time, step_dt)
    def reset(self):
        self.emulator.reset()
    def seed(self, seed):
        self.emulator.seed(seed)

class VirtualFlightController:
    def __init__(self, drone_mass, max_thrust):
        self.mass = float(drone_mass)
        self.max_thrust = float(max_thrust)
        self.hover_thrust = (self.mass * 9.81) / 4.0
        self.angle_to_rate_p = 3.0
        self.rate_kp, self.rate_ki, self.rate_kd = 0.15, 0.01, 0.005
        self.integrals = np.zeros(3, dtype=np.float64)
        self.prev_errors = np.zeros(3, dtype=np.float64)

    def reset(self):
        self.integrals[:] = 0.0
        self.prev_errors[:] = 0.0

    def compute_motors(self, target_roll, target_pitch, target_yaw_rate, thrust_norm, mj_data, dt, drone_body_id):
        target_roll = np.clip(target_roll, -0.6, 0.6)
        target_pitch = np.clip(target_pitch, -0.6, 0.6)
        target_yaw_rate = np.clip(target_yaw_rate, -1.5, 1.5)
        thrust_norm = np.clip(thrust_norm, 0.0, 1.0)
        quat = mj_data.xquat[drone_body_id]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler("xyz", degrees=False)
        roll_error = target_roll - euler[0]
        pitch_error = target_pitch - euler[1]
        desired_rates = np.array([roll_error * self.angle_to_rate_p, pitch_error * self.angle_to_rate_p, target_yaw_rate])
        current_rates = mj_data.qvel[3:6]
        rate_errors = desired_rates - current_rates
        torques = np.zeros(3, dtype=np.float64)
        for i in range(3):
            self.integrals[i] += rate_errors[i] * dt
            self.integrals[i] = np.clip(self.integrals[i], -5.0, 5.0)
            derivative = (rate_errors[i] - self.prev_errors[i]) / dt if dt > 0.0 else 0.0
            torque = self.rate_kp * rate_errors[i] + self.rate_ki * self.integrals[i] + self.rate_kd * derivative
            torques[i] = np.clip(torque, -5.0, 5.0)
            self.prev_errors[i] = rate_errors[i]
        roll_t, pitch_t, yaw_t = torques
        base_thrust = self.hover_thrust
        extra_thrust = (thrust_norm - 0.5) * 2.0 * self.hover_thrust
        attitude_comp = max(math.cos(euler[0]) * math.cos(euler[1]), 0.6)
        total_thrust = np.clip((base_thrust + extra_thrust) / attitude_comp, 0.0, self.max_thrust)
        m1 = total_thrust + roll_t + pitch_t - yaw_t
        m2 = total_thrust - roll_t + pitch_t + yaw_t
        m3 = total_thrust - roll_t - pitch_t - yaw_t
        m4 = total_thrust + roll_t - pitch_t + yaw_t
        return np.clip(np.array([m1, m2, m3, m4]), 0.0, self.max_thrust).astype(np.float32)

class Phase(str, Enum):
    TAKEOFF = "takeoff"
    HOVER = "hover"
    SEARCH = "search"
    PURSUIT = "pursuit"
    INTERCEPT = "intercept"
    MISSION = "mission"

class MissionStage:
    TAKEOFF = 0
    TRANSIT_1 = 1
    TRANSIT_2 = 2
    SEARCH = 3
    PURSUIT = 4
    RED_GATE = np.array([10.0, 0.0, 4.0])
    GREEN_GATE = np.array([22.0, 0.0, 8.0])

class DroneInterceptEnv(gym.Env):
    """
    Среда перехвата цели для переноса на реальное железо.
    
    Наблюдение (23 признака) — ТОЛЬКО то, что есть на реальном дроне:
       [0-4]   YOLO: dx, dy, visible, size, confidence (камера)
       [5-6]   Visual Velocity: v_dx, v_dy (оптический поток цели)
       [7-10]  Orientation: roll, pitch, sin_yaw, cos_yaw (IMU)
       [11-12] altitude, yaw_rate (barometer + IMU)
       [13-15] Memory: last_dx, last_dy, lost_time (память)
       [16-17] Lock: locked, lock_counter (статус захвата)
       [18-19] Drone GPS: x, y (GPS)
       [20-22] Optical Flow: flow_forward, flow_left, flow_right (камера)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    STATE_SIZE = 23  # 20 базовых + 3 оптического потока
    FLOW_MAX_DIST = 15.0  # Максимальная дальность "зрения" оптического потока

    def __init__(
        self,
        xml_path="scene.xml",
        phase=Phase.TAKEOFF,
        max_steps=1500,
        use_image=False,
        camera_resolution=(48, 48),
        action_repeat=4,
        detector=None,
        yolo_config=None,
    ):
        super().__init__()
        if isinstance(phase, str):
            phase = Phase(phase)
        self.phase = phase
        self.detector = detector if detector else SyntheticYOLODetector(cfg=yolo_config)
        self.use_image = bool(use_image)
        self.camera_resolution = camera_resolution
        self.max_steps = int(max_steps)
        self.action_repeat = int(action_repeat)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.step_dt = self.dt * self.action_repeat

        self.drone_id = self.model.body("x2").id
        self.target_id = self.model.body("target_kral").id
        self.target_mocap_id = self.model.body_mocapid[self.target_id]

        drone_mass = float(self.model.body_mass[self.drone_id])
        self.fc = VirtualFlightController(drone_mass=drone_mass, max_thrust=25.0)

        self.camera_fov_h = np.radians(100.0)
        self.camera_fov_v = np.radians(70.0)
        self.camera_max_distance = 60.0

        self.target_speed = 0.0
        self.target_vel = np.zeros(3, dtype=np.float64)
        self.target_maneuver_timer = 0.0
        self.target_maneuver_interval = 1.5
        self.grace_steps = 400

        # Для оптического потока
        self.prev_flow_distances = np.array([self.FLOW_MAX_DIST, self.FLOW_MAX_DIST, self.FLOW_MAX_DIST])

        self.action_space = spaces.Box(
            low=np.array([-0.6, -0.6, -1.5, 0.0], dtype=np.float32),
            high=np.array([0.6, 0.6, 1.5, 1.0], dtype=np.float32),
            dtype=np.float32)

        if self.use_image:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(0, 255, shape=(camera_resolution[0], camera_resolution[1], 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(self.STATE_SIZE,), dtype=np.float32)})
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.STATE_SIZE,), dtype=np.float32)

        self.renderer = None
        if self.use_image:
            self.renderer = mujoco.Renderer(self.model, height=camera_resolution[0], width=camera_resolution[1])

        self._reset_episode_state()
        logger.info("=" * 70)
        logger.info(f"DRONE INTERCEPT ENV — phase={self.phase.value.upper()}")
        logger.info(f"  detector={type(self.detector).__name__}, state_size={self.STATE_SIZE}")
        logger.info(f"  optical flow: 3 rays (forward, left, right), max_dist={self.FLOW_MAX_DIST}m")
        logger.info("=" * 70)

    def _reset_episode_state(self):
        self.current_step = 0
        self.sim_time = 0.0
        self.prev_target_size = 0.0
        self.last_dx, self.last_dy = 0.0, 0.0
        self.v_dx, self.v_dy = 0.0, 0.0
        self.last_confidence, self.lost_time = 0.0, 0.0
        self.search_direction = 1.0
        self.search_timer = 0.0
        self.search_switch_interval = 2.5
        self.lock_counter = 0
        self.locked = False
        self._takeoff_hold_steps = 0
        self._hover_hold_steps = 0
        self._pursuit_lock_steps = 0
        self._max_altitude_reached = 0.0
        self.mission_stage = MissionStage.TAKEOFF
        self.gate_1_passed = False
        self.gate_2_passed = False
        self._gate1_rewarded = False
        self._gate2_rewarded = False
        self.mission_target_ever_visible = False
        self._last_detection = Detection()
        self.prev_flow_distances = np.array([self.FLOW_MAX_DIST, self.FLOW_MAX_DIST, self.FLOW_MAX_DIST])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.detector.seed(seed)
        self.detector.reset()
        mujoco.mj_resetData(self.model, self.data)
        self._reset_episode_state()
        self.fc.reset()

        if self.phase == Phase.TAKEOFF:
            start_pos = [0.0, 0.0, 0.3]
            initial_yaw = self.np_random.uniform(-np.pi, np.pi)
        elif self.phase == Phase.HOVER:
            start_pos = [0.0, 0.0, 15.0]
            initial_yaw = self.np_random.uniform(-np.pi, np.pi)
        elif self.phase == Phase.SEARCH:
            start_pos = [0.0, 0.0, 8.0]
            initial_yaw = np.pi
        elif self.phase == Phase.PURSUIT:
            start_pos = [25.0, 0.0, 18.0]
            initial_yaw = self.np_random.uniform(-np.pi, np.pi)
        elif self.phase == Phase.INTERCEPT:
            start_pos = [32.0, 0.0, 16.0]
            initial_yaw = self.np_random.uniform(-np.pi, np.pi)
        else:
            start_pos = [0.0, 0.0, 0.3]
            initial_yaw = self.np_random.uniform(-np.pi, np.pi)

        self.data.qpos[0:3] = start_pos
        q = R.from_euler("z", initial_yaw).as_quat()
        self.data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        self.data.qvel[:] = 0.0

        if self.phase in (Phase.TAKEOFF, Phase.HOVER):
            target_pos = np.array([15.0, 0.0, 8.0])
            target_speed = 0.0
        elif self.phase == Phase.SEARCH:
            target_pos = np.array([38.0, 0.0, 16.0])
            target_speed = 0.0
        else:
            tx = self.np_random.uniform(28.0, 42.0)
            ty = self.np_random.uniform(-7.0, 7.0)
            tz = self.np_random.uniform(11.0, 17.0)
            target_pos = np.array([tx, ty, tz])
            target_speed = self.np_random.uniform(3.0, 5.0)

        self.data.mocap_pos[self.target_mocap_id] = target_pos
        self.target_speed = target_speed
        heading = self.np_random.uniform(-np.pi, np.pi)
        self.target_vel = np.array([
            self.target_speed * np.cos(heading),
            self.target_speed * np.sin(heading),
            0.0])
        self.target_maneuver_timer = 0.0
        self.search_direction = 1.0 if self.np_random.random() > 0.5 else -1.0

        mujoco.mj_forward(self.model, self.data)
        detection = self._detect_target()
        self._last_detection = detection
        if detection.visible:
            self.last_dx, self.last_dy = detection.dx, detection.dy

        return self._get_obs(detection), {}

    def step(self, action):
        self.current_step += 1
        self.sim_time += self.step_dt
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        roll, pitch, yaw_rate, thrust = float(action[0]), float(action[1]), float(action[2]), float(action[3])

        for _ in range(self.action_repeat):
            motors = self.fc.compute_motors(roll, pitch, yaw_rate, thrust, self.data, self.dt, self.drone_id)
            self.data.ctrl[0:4] = motors
            self._update_target()
            mujoco.mj_step(self.model, self.data)

        detection = self._detect_target()
        self._last_detection = detection

        # Визуальная скорость цели
        if detection.visible:
            self.v_dx = (detection.dx - self.last_dx) / self.step_dt if self.step_dt > 0 else 0.0
            self.v_dy = (detection.dy - self.last_dy) / self.step_dt if self.step_dt > 0 else 0.0
            self.last_dx, self.last_dy = detection.dx, detection.dy
        else:
            self.v_dx, self.v_dy = 0.0, 0.0

        if self.phase == Phase.MISSION:
            self._update_mission_stage()

        observation = self._get_obs(detection)
        reward, terminated, truncated, info = self._dispatch_reward(detection)
        return observation, reward, terminated, truncated, info

    def _detect_target(self):
        oracle = self._project_target_geometric()
        return self.detector.detect(oracle, self.sim_time, self.step_dt)

    def _project_target_geometric(self):
        drone_pos = self.data.xpos[self.drone_id]
        drone_quat = self.data.xquat[self.drone_id]
        rot = R.from_quat([drone_quat[1], drone_quat[2], drone_quat[3], drone_quat[0]])
        target_pos = self.data.mocap_pos[self.target_mocap_id]
        relative_world = target_pos - drone_pos
        distance = float(np.linalg.norm(relative_world))
        relative_body = rot.apply(relative_world, inverse=True)
        forward, right, up = float(relative_body[0]), float(relative_body[1]), float(relative_body[2])

        if distance > self.camera_max_distance or forward <= 0.1:
            return Detection(visible=False, distance=distance)
        if abs(np.arctan2(right, forward)) > self.camera_fov_h / 2.0:
            return Detection(visible=False, distance=distance)
        if abs(np.arctan2(up, forward)) > self.camera_fov_v / 2.0:
            return Detection(visible=False, distance=distance)

        dx = float(np.clip(right / (forward * np.tan(self.camera_fov_h / 2)), -1.0, 1.0))
        dy = float(np.clip(up / (forward * np.tan(self.camera_fov_v / 2)), -1.0, 1.0))
        size = float(np.clip(0.9 / max(forward, 0.1), 0.0, 1.0))
        return Detection(dx=dx, dy=dy, size=size, visible=True, confidence=1.0, distance=distance)

    def _get_optical_flow(self):
        """
        Имитирует оптический поток камеры (3 луча: forward, left, right).
        Возвращает нормализованную скорость приближения к объектам.
        """
        drone_pos = self.data.xpos[self.drone_id].astype(np.float64)
        quat = self.data.xquat[self.drone_id]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        
        # 3 направления: forward, left (30°), right (30°)
        angle = np.radians(30.0)
        directions = [
            rot.apply([1.0, 0.0, 0.0]),                    # forward
            rot.apply([np.cos(angle), -np.sin(angle), 0.0]),  # left
            rot.apply([np.cos(angle), np.sin(angle), 0.0]),   # right
        ]
        
        flow_values = []
        current_distances = []
        
        # Выходной массив для ID геометрии (перезаписывается при каждом вызове)
        geomid = np.zeros(1, dtype=np.int32)
        
        for direction in directions:
            direction = direction / np.linalg.norm(direction)
            direction = direction.astype(np.float64)
            
            # Правильный вызов mj_ray:
            # pnt - точка начала (float64)
            # vec - направление (float64)
            # geomgroup - None (проверять все группы)
            # flg_static - True (учитывать статичные объекты)
            # bodyexclude - -1 (не исключать никакие тела)
            # geomid - выходной массив для ID геометрии
            dist = mujoco.mj_ray(
                self.model, self.data,
                drone_pos, direction,
                None,   # geomgroup
                True,   # flg_static
                -1,     # bodyexclude
                geomid  # выходной массив
            )
            
            geom_id = int(geomid[0])
            
            if geom_id >= 0:
                geom_name = self.model.geom(geom_id).name
                # Игнорируем пол и саму цель
                if geom_name != "floor" and "target" not in geom_name:
                    current_distances.append(min(dist, self.FLOW_MAX_DIST))
                else:
                    current_distances.append(self.FLOW_MAX_DIST)
            else:
                current_distances.append(self.FLOW_MAX_DIST)
        
        # Оптический поток = скорость изменения расстояния
        for i in range(3):
            prev_dist = self.prev_flow_distances[i]
            curr_dist = current_distances[i]
            flow = (prev_dist - curr_dist) / self.step_dt if self.step_dt > 0 else 0.0
            flow_norm = float(np.clip(flow / 10.0, -1.0, 1.0))
            flow_values.append(flow_norm)
        
        self.prev_flow_distances = np.array(current_distances)
        return tuple(flow_values)  # (flow_forward, flow_left, flow_right)
        
    def _update_mission_stage(self):
        drone_pos = self.data.xpos[self.drone_id]
        altitude = float(drone_pos[2])
        
        if self.mission_stage == MissionStage.TAKEOFF and altitude > 4.0:
            self.mission_stage = MissionStage.TRANSIT_1
        
        if (self.mission_stage == MissionStage.TRANSIT_1
                and drone_pos[0] > MissionStage.RED_GATE[0]
                and abs(drone_pos[1]) < 4.0):
            self.mission_stage = MissionStage.TRANSIT_2
            self.gate_1_passed = True
        
        if (self.mission_stage == MissionStage.TRANSIT_2
                and drone_pos[0] > MissionStage.GREEN_GATE[0]
                and abs(drone_pos[1]) < 4.0):
            self.mission_stage = MissionStage.SEARCH
            self.gate_2_passed = True
        
        if (self.mission_stage == MissionStage.SEARCH
                and self._last_detection.visible):
            self.mission_stage = MissionStage.PURSUIT
            self.mission_target_ever_visible = True

    def _get_obs(self, detection):
        if detection.visible:
            self.last_confidence = detection.confidence
            self.lost_time = 0.0
        else:
            self.lost_time += self.step_dt

        if (detection.visible and abs(detection.dx) < 0.15 and abs(detection.dy) < 0.15
                and detection.size > 0.01 and detection.confidence > 0.5):
            self.lock_counter += 1
        else:
            self.lock_counter = 0
        self.locked = self.lock_counter >= 8

        if not detection.visible:
            self.search_timer += self.step_dt
            if self.search_timer >= self.search_switch_interval:
                self.search_timer = 0.0
                self.search_direction *= -1.0
        else:
            self.search_timer = 0.0

        drone_pos = self.data.xpos[self.drone_id]
        euler = self._get_euler()
        altitude = float(drone_pos[2])
        yaw_rate = float(self.data.qvel[5])

        v_dx_norm = float(np.clip(self.v_dx / 10.0, -1.0, 1.0))
        v_dy_norm = float(np.clip(self.v_dy / 10.0, -1.0, 1.0))
        
        # Оптический поток (3 признака)
        flow_forward, flow_left, flow_right = self._get_optical_flow()

        state = np.array([
            # [0-4] YOLO
            detection.dx, detection.dy, float(detection.visible),
            np.clip(detection.size, 0.0, 1.0), np.clip(detection.confidence, 0.0, 1.0),
            # [5-6] Visual Velocity
            v_dx_norm, v_dy_norm,
            # [7-10] Orientation
            np.clip(euler[0] / 1.5, -1.0, 1.0), np.clip(euler[1] / 1.5, -1.0, 1.0),
            np.sin(euler[2]), np.cos(euler[2]),
            # [11-12] Dynamics
            np.clip(altitude / 20.0, 0.0, 1.5), np.clip(yaw_rate / 1.5, -1.0, 1.0),
            # [13-15] Memory
            self.last_dx, self.last_dy, np.clip(self.lost_time / 3.0, 0.0, 1.0),
            # [16-17] Lock Status
            float(self.locked), np.clip(self.lock_counter / 10.0, 0.0, 1.0),
            # [18-19] Drone GPS
            float(np.clip(drone_pos[0] / 45.0, 0.0, 1.0)), float(np.clip(drone_pos[1] / 10.0, -1.0, 1.0)),
            # [20-22] Optical Flow (НОВОЕ)
            flow_forward, flow_left, flow_right,
        ], dtype=np.float32)

        self.prev_target_size = float(detection.size)

        if self.use_image:
            return {"image": self._render_camera(), "state": state}
        return state

    def _get_euler(self):
        quat = self.data.xquat[self.drone_id]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rot.as_euler("xyz")

    def _render_camera(self):
        self.renderer.update_scene(self.data, camera="fpv_cam")
        return np.asarray(self.renderer.render(), dtype=np.uint8)

    def _update_target(self):
        if self.phase not in (Phase.PURSUIT, Phase.INTERCEPT, Phase.MISSION):
            return
        self.target_maneuver_timer += self.dt
        if self.target_maneuver_timer >= self.target_maneuver_interval:
            self.target_maneuver_timer = 0.0
            current_speed = float(np.linalg.norm(self.target_vel[:2]))
            current_heading = float(np.arctan2(self.target_vel[1], self.target_vel[0]))
            new_heading = current_heading + float(self.np_random.uniform(-np.pi / 2, np.pi / 2))
            vertical_velocity = float(self.np_random.uniform(-0.6, 0.6))
            self.target_vel = np.array([
                current_speed * np.cos(new_heading),
                current_speed * np.sin(new_heading),
                vertical_velocity])

        self.data.mocap_pos[self.target_mocap_id] += self.target_vel * self.dt
        pos = self.data.mocap_pos[self.target_mocap_id]
        if pos[0] <= 25.0:   pos[0] = 25.0; self.target_vel[0] = abs(self.target_vel[0])
        elif pos[0] >= 45.0: pos[0] = 45.0; self.target_vel[0] = -abs(self.target_vel[0])
        if pos[1] <= -8.0:   pos[1] = -8.0; self.target_vel[1] = abs(self.target_vel[1])
        elif pos[1] >= 8.0:  pos[1] = 8.0;  self.target_vel[1] = -abs(self.target_vel[1])
        if pos[2] <= 10.0:   pos[2] = 10.0; self.target_vel[2] = abs(self.target_vel[2])
        elif pos[2] >= 18.0: pos[2] = 18.0; self.target_vel[2] = -abs(self.target_vel[2])

    def _check_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name
            if g1 == "floor" or g2 == "floor": continue
            b1, b2 = self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]
            if b1 == self.drone_id or b2 == self.drone_id: return True
        return False

    def _altitude_penalty_soft(self):
        x, z = float(self.data.xpos[self.drone_id][0]), float(self.data.xpos[self.drone_id][2])
        safe_alt = 0.5 if x < 2.0 else (4.0 if x < 10.0 else (3.0 if x < 25.0 else 9.0))
        deficit = safe_alt - z
        return -float(np.clip(1.0 * deficit, 0.0, 10.0)) if deficit > 0 else 0.0

    def _dispatch_reward(self, detection):
        dispatch = {
            Phase.TAKEOFF: self._reward_takeoff,
            Phase.HOVER: self._reward_hover,
            Phase.SEARCH: self._reward_search,
            Phase.PURSUIT: self._reward_pursuit,
            Phase.INTERCEPT: self._reward_intercept,
            Phase.MISSION: self._reward_mission,
        }
        return dispatch[self.phase](detection)

    def _reward_takeoff(self, detection):
        altitude = float(self.data.xpos[self.drone_id][2])
        euler = self._get_euler()
        tilt = euler[0] ** 2 + euler[1] ** 2
        z_vel = float(self.data.qvel[2])
        reward = 0.0
        target_alt = 5.0
        hold_threshold = 4.5

        reward += 5.0 * min(altitude, 5.0)
        reward += 2.0 * np.clip(z_vel, 0.0, 3.0)
        if abs(euler[0]) > 1.0 or abs(euler[1]) > 1.0:
            reward -= 1.0
        reward -= 0.001

        if altitude >= hold_threshold:
            self._takeoff_hold_steps += 1
            reward += 2.0
        else:
            self._takeoff_hold_steps = 0

        if self._takeoff_hold_steps >= 15:
            return reward + 100.0, True, False, {"is_success": True, "phase": "takeoff", "altitude": altitude}

        if self.current_step > 400:
            if abs(euler[0]) > 1.4 or abs(euler[1]) > 1.4:
                return reward - 5.0, True, False, {"is_success": False, "reason": "crash_attitude"}
            if self._check_collision():
                return reward - 5.0, True, False, {"is_success": False, "reason": "crash_obstacle"}

        return reward, False, self.current_step >= self.max_steps, {"altitude": altitude}

    def _reward_hover(self, detection):
        altitude = float(self.data.xpos[self.drone_id][2])
        euler = self._get_euler()
        z_vel = float(self.data.qvel[2])
        target_alt = 15.0
        alt_error = altitude - target_alt
        tilt = euler[0] ** 2 + euler[1] ** 2
        reward = 0.0

        reward += 3.0 * np.exp(-0.05 * alt_error ** 2)
        reward += 1.0 * np.clip(1.0 - abs(alt_error) / target_alt, 0.0, 1.0)
        reward -= 0.3 * np.clip(abs(z_vel), 0.0, 2.0)
        reward -= 0.3 * tilt
        if abs(euler[0]) > 0.8 or abs(euler[1]) > 0.8:
            reward -= 0.5
        reward -= 0.005

        if abs(alt_error) < 0.5 and tilt < 0.2:
            self._hover_hold_steps += 1
        else:
            self._hover_hold_steps = 0

        if self._hover_hold_steps >= 300:
            return reward + 30.0, True, False, {"is_success": True, "phase": "hover", "altitude": altitude}

        if self.current_step > self.grace_steps:
            if altitude < 2.0 or abs(euler[0]) > 1.3 or abs(euler[1]) > 1.3 or self._check_collision():
                return reward - 10.0, True, False, {"is_success": False, "reason": "crash", "altitude": altitude}

        return reward, False, self.current_step >= self.max_steps, {"altitude": altitude, "tilt": tilt}

    def _reward_search(self, detection):
        altitude = float(self.data.xpos[self.drone_id][2])
        yaw_rate = float(self.data.qvel[5])
        alt_error = altitude - 8.0
        reward = 0.0

        reward += 0.3 * np.exp(-0.15 * alt_error ** 2)
        if detection.visible:
            reward += 10.0
            centering = np.sqrt(detection.dx ** 2 + detection.dy ** 2)
            reward += 0.5 * (1.0 - np.clip(centering, 0.0, 1.4))
            reward += 1.0 * detection.confidence
            if self.current_step > 50:
                return reward, True, False, {"is_success": True, "phase": "search", "visible": True}
        else:
            abs_yaw = abs(yaw_rate)
            if abs_yaw > 0.8:
                reward += 0.4
            elif abs_yaw > 0.4:
                reward += 0.15
            else:
                reward -= 0.15

        if self.current_step > self.grace_steps:
            if altitude < 1.0 or self._check_collision():
                return reward - 10.0, True, False, {"is_success": False, "reason": "crash", "visible": False}

        reward -= 0.01
        return reward, False, self.current_step >= self.max_steps, {"visible": detection.visible, "yaw_rate": yaw_rate}

    def _reward_pursuit(self, detection):
        distance = detection.distance
        reward = 0.0

        reward += 2.0 * np.exp(-0.05 * distance)
        if detection.visible:
            centering = np.sqrt(detection.dx ** 2 + detection.dy ** 2)
            reward += 1.0 * (1.0 - np.clip(centering, 0.0, 1.4))
            reward += 0.5 * detection.confidence
            reward += 4.0 * detection.size
            visual_speed = abs(self.v_dx) + abs(self.v_dy)
            reward -= 1.0 * np.clip(visual_speed / 5.0, 0.0, 1.0)

        if distance < 5.0 and detection.visible:
            self._pursuit_lock_steps += 1
        else:
            self._pursuit_lock_steps = 0

        if self._pursuit_lock_steps >= 30:
            return reward + 30.0, True, False, {"is_success": True, "phase": "pursuit", "distance": distance, "visible": True}

        if not detection.visible:
            reward -= 0.3

        if self.current_step > self.grace_steps:
            if self._check_collision():
                return reward - 20.0, True, False, {"is_success": False, "reason": "crash_obstacle", "distance": distance}

        reward -= 0.01
        return reward, False, self.current_step >= self.max_steps, {"distance": distance, "visible": detection.visible}

    def _reward_intercept(self, detection):
        distance = detection.distance
        reward = 0.0

        reward += 2.0 * np.exp(-0.1 * distance)
        if detection.visible:
            centering = np.sqrt(detection.dx ** 2 + detection.dy ** 2)
            reward += 1.0 * (1.0 - np.clip(centering, 0.0, 1.4))
            reward += 0.5 * detection.confidence
        reward -= 0.01

        if distance < 2.0:
            return reward + 200.0, True, False, {"is_success": True, "phase": "intercept", "distance": distance, "visible": detection.visible}

        if not detection.visible:
            reward -= 1.0

        if self.current_step > self.grace_steps:
            if self._check_collision():
                return reward - 20.0, True, False, {"is_success": False, "reason": "crash", "distance": distance}

        return reward, False, self.current_step >= self.max_steps, {"distance": distance, "visible": detection.visible}

    def _reward_mission(self, detection):
        drone_pos = self.data.xpos[self.drone_id]
        altitude = float(drone_pos[2])
        euler = self._get_euler()
        tilt = euler[0] ** 2 + euler[1] ** 2
        distance = detection.distance
        reward = 0.0

        if self.mission_stage in (MissionStage.TAKEOFF, MissionStage.TRANSIT_1, MissionStage.TRANSIT_2):
            reward += self._altitude_penalty_soft()

        if self.mission_stage == MissionStage.TAKEOFF:
            reward += 5.0 * min(altitude, 5.0)
        elif self.mission_stage == MissionStage.TRANSIT_1:
            dist_to_gate = float(np.linalg.norm(drone_pos - MissionStage.RED_GATE))
            reward += 2.0 * np.exp(-0.05 * dist_to_gate)
            reward += 1.0 * np.clip(drone_pos[0] / MissionStage.RED_GATE[0], 0.0, 1.0)
            # НОВОЕ: Бонус за пролёт через ворота (одноразовый)
            if drone_pos[0] > MissionStage.RED_GATE[0] + 1.0 and not self._gate1_rewarded:
                reward += 20.0
                self._gate1_rewarded = True
        elif self.mission_stage == MissionStage.TRANSIT_2:
            dist_to_gate = float(np.linalg.norm(drone_pos - MissionStage.GREEN_GATE))
            reward += 2.0 * np.exp(-0.05 * dist_to_gate)
            reward += 1.0 * np.clip(drone_pos[0] / MissionStage.GREEN_GATE[0], 0.0, 1.0)
            # НОВОЕ: Бонус за пролёт через ворота (одноразовый)
            if drone_pos[0] > MissionStage.GREEN_GATE[0] + 1.0 and not self._gate2_rewarded:
                reward += 20.0
                self._gate2_rewarded = True
        elif self.mission_stage == MissionStage.SEARCH:
            if detection.visible:
                reward += 10.0
            else:
                abs_yaw = abs(float(self.data.qvel[5]))
                if abs_yaw > 0.4:
                    reward += 0.3
            reward += 1.0 * np.exp(-0.03 * distance)
        elif self.mission_stage == MissionStage.PURSUIT:
            reward += 2.0 * np.exp(-0.1 * distance)
            if detection.visible:
                centering = np.sqrt(detection.dx ** 2 + detection.dy ** 2)
                reward += 1.0 * (1.0 - np.clip(centering, 0.0, 1.4))
                reward += 0.5 * detection.confidence
                reward += 4.0 * detection.size
                visual_speed = abs(self.v_dx) + abs(self.v_dy)
                reward -= 1.0 * np.clip(visual_speed / 5.0, 0.0, 1.0)

        # НОВОЕ: Штраф за оптический поток (избегание столкновений)
        flow_forward, flow_left, flow_right = self._get_optical_flow()
        # Штраф за приближение к препятствиям впереди
        if flow_forward > 0.3:  # Объекты быстро приближаются
            reward -= 2.0 * flow_forward
        # Штраф за приближение к препятствиям слева/справа
        if abs(flow_left) > 0.3 or abs(flow_right) > 0.3:
            reward -= 1.0 * max(abs(flow_left), abs(flow_right))

        if detection.visible and not self.mission_target_ever_visible:
            reward += 15.0

        if distance < 2.0:
            return reward + 200.0, True, False, {"is_success": True, "phase": "mission", "distance": distance, "stage": self.mission_stage}

        reward -= 0.2 * tilt
        if abs(euler[0]) > 0.9 or abs(euler[1]) > 0.9:
            reward -= 0.5
        reward -= 0.005

        if self.current_step > self.grace_steps:
            if abs(euler[0]) > 1.3 or abs(euler[1]) > 1.3:
                return reward - 15.0, True, False, {"is_success": False, "reason": "crash_attitude", "stage": self.mission_stage}
            if self._check_collision():
                return reward - 20.0, True, False, {"is_success": False, "reason": "crash_obstacle", "stage": self.mission_stage}
            if altitude < 0.5 and self.mission_stage > MissionStage.TAKEOFF:
                return reward - 15.0, True, False, {"is_success": False, "reason": "ground_crash", "stage": self.mission_stage}

        return reward, False, self.current_step >= self.max_steps, {"distance": distance, "stage": self.mission_stage, "visible": detection.visible}

    def render(self):
        return self._render_camera() if self.use_image else None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()