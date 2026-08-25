import os
import sys
import argparse
import time
from typing import Optional

# ============================================================
# CPU ONLY НАСТРОЙКИ
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
CPU_THREADS = max(1, (os.cpu_count() or 4) - 1)

for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = str(CPU_THREADS)

# ============================================================
# IMPORTS
# ============================================================
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from drone_intercept_env import (
    DroneInterceptEnv,
    Phase,
    OracleDetector,
    SyntheticYOLODetector,
    YOLOEmulatorConfig,
)

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "./models"
LOG_DIR = "./rl_logs"
CHECKPOINT_DIR = "./checkpoints"

CHECKPOINT_FREQ = 50_000  # Чекпоинты каждые 50k шагов для экономии места
NUM_ENVS = 4

# Используем единую фазу MISSION для сквозного обучения (взлёт -> транзит -> перехват)
CURRICULUM = [
    (Phase.MISSION, 800_000, 2000, 10_000),
]

TOTAL_TIMESTEPS = sum(steps for _, steps, _, _ in CURRICULUM)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# АРГУМЕНТЫ
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Drone Intercept Training (Realistic)")
    parser.add_argument("--detector", type=str, default="synthetic_yolo", choices=["oracle", "synthetic_yolo"])
    parser.add_argument("--noise-level", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--start-phase", type=int, default=0)
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    return parser.parse_args()

def get_yolo_config(noise_level: str) -> YOLOEmulatorConfig:
    configs = {
        "low": YOLOEmulatorConfig(base_false_negative_rate=0.03, bbox_std_fraction=0.02, latency_seconds=0.03, confidence_threshold=0.35),
        "medium": YOLOEmulatorConfig(base_false_negative_rate=0.08, bbox_std_fraction=0.04, latency_seconds=0.06, confidence_threshold=0.40),
        "high": YOLOEmulatorConfig(base_false_negative_rate=0.15, bbox_std_fraction=0.06, latency_seconds=0.10, confidence_threshold=0.45),
    }
    return configs[noise_level]

# ============================================================
# CALLBACK
# ============================================================
class FullCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, vec_env, phase_name, verbose=1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.vec_env = vec_env
        self.phase_name = phase_name

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step = int(self.num_timesteps)
            model_path = os.path.join(self.save_path, f"{self.phase_name}_checkpoint_{step}.zip")
            norm_path = os.path.join(self.save_path, f"{self.phase_name}_vec_normalize_{step}.pkl")
            self.model.save(model_path)
            self.vec_env.save(norm_path)
            print(f"\n{'='*70}\n[CHECKPOINT] Phase: {self.phase_name.upper()} | {step:,} steps\n{'='*70}\n")
        return True

# ============================================================
# ENV FACTORY
# ============================================================
def make_env(phase: Phase, max_steps: int, detector_type: str, yolo_config: Optional[YOLOEmulatorConfig]):
    def _init():
        detector = OracleDetector() if detector_type == "oracle" else SyntheticYOLODetector(cfg=yolo_config)
        env = DroneInterceptEnv(
            xml_path="scene.xml",
            phase=phase,
            max_steps=max_steps,
            use_image=False,
            detector=detector,
        )
        return Monitor(env)
    return _init

# ============================================================
# VALIDATION
# ============================================================
def validate_phase(phase: Phase, model: SAC, detector_type: str, yolo_config: Optional[YOLOEmulatorConfig], n_episodes: int = 10) -> dict:
    print(f"  [VALIDATION] Running {n_episodes} episodes...")
    env = DummyVecEnv([make_env(phase, 3000, detector_type, yolo_config)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env.training = False

    successes, total_rewards, ep_lengths = 0, [], []
    for _ in range(n_episodes):
        obs = env.reset()
        ep_reward, ep_len, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1
            if done[0]:
                if info[0].get("is_success", False): successes += 1
                break
        total_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
    env.close()

    result = {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_ep_len": float(np.mean(ep_lengths)),
    }
    print(f"  [VALIDATION] success_rate={result['success_rate']:.0%} | mean_reward={result['mean_reward']:.1f} | mean_ep_len={result['mean_ep_len']:.0f}")
    return result

# ============================================================
# PRINT CONFIG
# ============================================================
def print_configuration(args):
    print("\n" + "="*70)
    print("DRONE INTERCEPT TRAINING (REALISTIC HARDWARE-READY)")
    print("="*70 + "\n")
    print("OBSERVATION (23 features - NO CHEATING):")
    print("  [0-4]   YOLO: dx, dy, visible, size, confidence")
    print("  [5-6]   Visual Velocity: v_dx, v_dy (target motion in frame)")
    print("  [7-10]  Orientation: roll, pitch, sin_yaw, cos_yaw (IMU)")
    print("  [11-12] Dynamics: altitude, yaw_rate (Baro + IMU)")
    print("  [13-15] Memory: last_dx, last_dy, lost_time")
    print("  [16-17] Lock Status: locked, lock_counter")
    print("  [18-19] Drone GPS: x, y")
    print("  [20-22] Optical Flow: flow_forward, flow_left, flow_right (Obstacle avoidance)")
    print(f"\nDETECTOR: {args.detector.upper()} (Noise: {args.noise_level.upper()})")
    print(f"TOTAL STEPS : {TOTAL_TIMESTEPS:,}")
    print(f"PARALLEL ENV: {NUM_ENVS}")
    print("="*70 + "\n")

# ============================================================
# TRAIN
# ============================================================
def train():
    args = parse_args()
    print_configuration(args)

    torch.set_num_threads(CPU_THREADS)
    yolo_config = get_yolo_config(args.noise_level) if args.detector == "synthetic_yolo" else None

    model = None
    validation_results = {}

    for phase_idx, (phase, total_steps, ep_steps, learning_starts) in enumerate(CURRICULUM):
        if phase_idx < args.start_phase:
            continue

        print(f"\n{'='*70}\nPHASE: {phase.value.upper()}\n{'='*70}\n")

        raw_env = DummyVecEnv([make_env(phase, ep_steps, args.detector, yolo_config) for _ in range(NUM_ENVS)])
        vec_env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=30.0, clip_reward=5.0)

        phase_ckpt_dir = os.path.join(CHECKPOINT_DIR, phase.value)
        os.makedirs(phase_ckpt_dir, exist_ok=True)
        
        checkpoint_callback = FullCheckpointCallback(
            save_freq=max(1, CHECKPOINT_FREQ // NUM_ENVS),
            save_path=phase_ckpt_dir,
            vec_env=vec_env,
            phase_name=phase.value,
        )

        if model is None:
            print("  Creating new SAC model...")
            model = SAC(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=3e-4,
                buffer_size=500_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                learning_starts=learning_starts,
                ent_coef="auto",
                target_entropy=-2.0,  # ВАЖНО: предотвращает схлопывание энтропии
                policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh),
                device="cpu",
                verbose=1,
                tensorboard_log=LOG_DIR,
            )
        else:
            model.set_env(vec_env)
            model.replay_buffer.reset()

        phase_start_time = time.time()
        model.learn(
            total_timesteps=total_steps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            log_interval=10,
            progress_bar=True,
        )
        print(f"  [TIME] Phase completed in {(time.time() - phase_start_time)/60:.1f} minutes")

        final_model_path = os.path.join(MODEL_DIR, f"sac_{phase.value}")
        final_norm_path = os.path.join(MODEL_DIR, f"vec_norm_{phase.value}.pkl")
        model.save(final_model_path)
        vec_env.save(final_norm_path)

        if args.validate:
            val_result = validate_phase(phase, model, args.detector, yolo_config, n_episodes=10)
            validation_results[phase.value] = val_result

        vec_env.close()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    if validation_results:
        for phase_name, result in validation_results.items():
            print(f"  {phase_name.upper():12s} | success={result['success_rate']:.0%} | reward={result['mean_reward']:.1f}")

if __name__ == "__main__":
    train()