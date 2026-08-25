import os
import sys
import argparse
import time

# ============================================================
# CPU
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = ""

CPU_THREADS = max(
    1,
    (os.cpu_count() or 4) - 1
)

for key in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS"
]:

    os.environ[key] = str(
        CPU_THREADS
    )


# ============================================================
# IMPORTS
# ============================================================

import torch
import numpy as np

from stable_baselines3 import SAC

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize
)

from stable_baselines3.common.callbacks import (
    BaseCallback
)

from stable_baselines3.common.monitor import (
    Monitor
)

from drone_intercept_env import (
    DroneInterceptEnv,
    Phase,
    CameraTargetDetector
)


# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR = "./models"

LOG_DIR = "./rl_logs"

CHECKPOINT_DIR = "./checkpoints"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIG
# ============================================================

NUM_ENVS = 4

TOTAL_TIMESTEPS = 800_000

MAX_EPISODE_STEPS = 2000

LEARNING_STARTS = 10_000

CHECKPOINT_FREQ = 50_000


# ============================================================
# SEED
# ============================================================

SEED = 42


# ============================================================
# CALLBACK
# ============================================================

class FullCheckpointCallback(
    BaseCallback
):

    def __init__(
        self,
        save_freq,
        save_path,
        vec_env,
        verbose=1
    ):

        super().__init__(
            verbose
        )

        self.save_freq = int(
            save_freq
        )

        self.save_path = (
            save_path
        )

        self.vec_env = vec_env

    def _on_step(self):

        if (
            self.n_calls %
            self.save_freq
            == 0
        ):

            step = int(
                self.num_timesteps
            )

            model_path = os.path.join(
                self.save_path,
                f"sac_camera_{step}"
            )

            norm_path = os.path.join(
                self.save_path,
                f"vecnorm_camera_{step}.pkl"
            )

            self.model.save(
                model_path
            )

            self.vec_env.save(
                norm_path
            )

            print()
            print("=" * 70)
            print(
                f"CHECKPOINT: {step:,}"
            )
            print("=" * 70)

        return True


# ============================================================
# ENV FACTORY
# ============================================================

def make_env(
    rank,
    seed=SEED
):

    def _init():

        detector = CameraTargetDetector(
            image_width=96,
            image_height=96,
            noise_level="medium"
        )

        env = DroneInterceptEnv(

            xml_path="scene.xml",

            phase=Phase.MISSION,

            max_steps=MAX_EPISODE_STEPS,

            use_image=False,

            camera_resolution=(
                96,
                96
            ),

            action_repeat=4,

            detector=detector
        )

        env = Monitor(
            env
        )

        env.reset(
            seed=seed + rank
        )

        return env

    return _init


# ============================================================
# CREATE ENV
# ============================================================

def create_training_env():

    env = DummyVecEnv(

        [
            make_env(
                rank=i
            )

            for i in range(
                NUM_ENVS
            )
        ]

    )

    env = VecNormalize(

        env,

        norm_obs=True,

        # Для reward лучше НЕ нормализовать,
        # потому что мы хотим видеть реальный reward.
        norm_reward=False,

        clip_obs=10.0

    )

    return env


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    vec_env,
    episodes=20
):

    print()
    print("=" * 70)
    print(
        "VALIDATION"
    )
    print("=" * 70)

    # ========================================================
    # Очень важно:
    # использовать те же statistics,
    # которые были получены во время обучения.
    # ========================================================

    vec_env.training = False

    vec_env.norm_reward = False

    successes = 0

    rewards = []

    lengths = []

    collisions = 0

    reasons = {}

    for episode in range(
        episodes
    ):

        obs = vec_env.reset()

        total_reward = 0.0

        length = 0

        while True:

            action, _ = model.predict(

                obs,

                deterministic=True
            )

            obs, reward, done, infos = (
                vec_env.step(
                    action
                )
            )

            total_reward += float(
                reward[0]
            )

            length += 1

            if done[0]:

                info = infos[0]

                if info.get(
                    "is_success",
                    False
                ):

                    successes += 1

                reason = info.get(
                    "reason",
                    "finished"
                )

                reasons[reason] = (
                    reasons.get(
                        reason,
                        0
                    )
                    + 1
                )

                if reason == (
                    "crash_obstacle"
                ):

                    collisions += 1

                break

        rewards.append(
            total_reward
        )

        lengths.append(
            length
        )

    success_rate = (
        successes /
        max(episodes, 1)
    )

    collision_rate = (
        collisions /
        max(episodes, 1)
    )

    print()
    print(
        f"Success rate : "
        f"{success_rate:.1%}"
    )

    print(
        f"Mean reward  : "
        f"{np.mean(rewards):.2f}"
    )

    print(
        f"Mean length  : "
        f"{np.mean(lengths):.1f}"
    )

    print(
        f"Collision    : "
        f"{collision_rate:.1%}"
    )

    print()
    print(
        "Termination reasons:"
    )

    for reason, count in (
        sorted(
            reasons.items(),
            key=lambda x: -x[1]
        )
    ):

        print(
            f"  {reason:25s}"
            f"{count}"
        )

    print(
        "=" * 70
    )

    return {
        "success_rate":
            success_rate,

        "mean_reward":
            float(
                np.mean(
                    rewards
                )
            ),

        "mean_length":
            float(
                np.mean(
                    lengths
                )
            ),

        "collision_rate":
            collision_rate,

        "reasons":
            reasons
    }


# ============================================================
# TRAIN
# ============================================================

def train():

    print()
    print("=" * 70)
    print(
        "CAMERA-ONLY DRONE INTERCEPT"
    )
    print("=" * 70)

    print()
    print(
        "NO LIDAR"
    )

    print(
        "NO RAYCAST"
    )

    print(
        "NO DEPTH"
    )

    print(
        "NO TARGET DISTANCE"
    )

    print(
        "NO OBSTACLE DISTANCE"
    )

    print()
    print(
        f"CPU threads : "
        f"{CPU_THREADS}"
    )

    print(
        f"Parallel env: "
        f"{NUM_ENVS}"
    )

    print(
        f"Timesteps   : "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        "=" * 70
    )

    torch.set_num_threads(
        CPU_THREADS
    )

    # ========================================================
    # ENV
    # ========================================================

    vec_env = (
        create_training_env()
    )

    # ========================================================
    # SAC
    # ========================================================

    print()
    print(
        "Creating SAC..."
    )

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

        learning_starts=
            LEARNING_STARTS,

        ent_coef="auto",

        target_entropy=-2.0,

        policy_kwargs=dict(

            net_arch=[
                256,
                256
            ],

            activation_fn=
                torch.nn.Tanh
        ),

        device="cpu",

        verbose=1,

        tensorboard_log=
            LOG_DIR,

        seed=SEED
    )

    # ========================================================
    # CALLBACK
    # ========================================================

    callback = (
        FullCheckpointCallback(

            save_freq=max(
                1,
                CHECKPOINT_FREQ //
                NUM_ENVS
            ),

            save_path=
                CHECKPOINT_DIR,

            vec_env=
                vec_env
        )
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print(
        "START TRAINING"
    )

    start_time = time.time()

    model.learn(

        total_timesteps=
            TOTAL_TIMESTEPS,

        callback=callback,

        reset_num_timesteps=True,

        log_interval=10,

        progress_bar=True
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    print()
    print(
        f"Training time: "
        f"{elapsed / 60:.1f} min"
    )

    # ========================================================
    # SAVE
    # ========================================================

    model_path = os.path.join(
        MODEL_DIR,
        "sac_camera_only"
    )

    vecnorm_path = os.path.join(
        MODEL_DIR,
        "vecnorm_camera_only.pkl"
    )

    model.save(
        model_path
    )

    vec_env.save(
        vecnorm_path
    )

    print()
    print(
        "MODEL SAVED:"
    )

    print(
        model_path
    )

    print(
        vecnorm_path
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    result = validate(

        model,

        vec_env,

        episodes=20
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Success: "
        f"{result['success_rate']:.1%}"
    )

    print(
        f"Reward : "
        f"{result['mean_reward']:.2f}"
    )

    print(
        f"Crash  : "
        f"{result['collision_rate']:.1%}"
    )

    print("=" * 70)

    vec_env.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()