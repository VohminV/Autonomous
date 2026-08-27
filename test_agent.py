# ============================================================
# test_visual_follow.py
#
# TEST CURRENT VisualFollowEnv / SAC MODEL
#
# V16.3
#
# Observation:
#   image = 48 x 48 x 3
#   state = 11
#
# Action:
#   [-1,+1] x 4
#
# Model:
#   visual_follow.zip
#
# DEBUG ONLY:
#   true distance
#   target velocity
#   target position
#   drone position
#   visual projection
#
# Эти данные НЕ передаются агенту.
# ============================================================

from __future__ import annotations

import os
import time
import math
import argparse
from datetime import datetime

import cv2
import numpy as np

from stable_baselines3 import SAC

from train_rl_agent import (
    VisualFollowEnv,
    CFG,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Test VisualFollow SAC agent"
    )

    parser.add_argument(
        "--model",
        default="visual_follow.zip",
        help="Путь к SAC модели",
    )

    parser.add_argument(
        "--xml",
        default="scene.xml",
        help="MuJoCo XML",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Количество тестовых эпизодов",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=800,
        help="Максимум шагов на эпизод",
    )

    parser.add_argument(
        "--output",
        default="test_videos",
        help="Каталог видео",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Не сохранять MP4",
    )

    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Не показывать окно",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Использовать stochastic policy вместо deterministic",
    )

    parser.add_argument(
        "--phase",
        type=int,
        default=4,
        help="Curriculum phase to test in (1-4)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Начальный seed",
    )

    return parser.parse_args()


# ============================================================
# VISUAL INFO
# ============================================================

class VisualInfo:
    """
    DEBUG-only визуальная информация.

    Ничего из этого класса не передаётся SAC.
    """

    def __init__(self, env):

        self.visible = bool(
            getattr(
                env,
                "last_target_visible",
                False,
            )
        )

        self.confidence = float(
            getattr(
                env,
                "last_visual_score",
                0.0,
            )
        )

        self.dx = 0.0
        self.dy = 0.0
        self.size = 0.0

        if (
            self.visible
            and env.target_body_id is not None
        ):

            drone_pos = env.data.xpos[
                env.drone_body_id
            ]

            target_pos = env.data.xpos[
                env.target_body_id
            ]

            rel = (
                target_pos
                - drone_pos
            )

            rot = env.data.xmat[
                env.drone_body_id
            ].reshape(3, 3)

            local_rel = rot.T @ rel

            if local_rel[0] > 0.1:

                fov_rad = math.radians(
                    CFG.camera_fov
                )

                f = (
                    1.0
                    / math.tan(
                        fov_rad / 2.0
                    )
                )

                self.dx = (
                    -(local_rel[1]
                      / local_rel[0])
                    * f
                )

                self.dy = (
                    -(local_rel[2]
                      / local_rel[0])
                    * f
                )

                self.dx = max(
                    -1.0,
                    min(1.0, self.dx),
                )

                self.dy = max(
                    -1.0,
                    min(1.0, self.dy),
                )

                distance = np.linalg.norm(
                    rel
                )

                self.size = max(
                    0.05,
                    min(
                        1.0,
                        2.0
                        / max(
                            distance,
                            1e-3,
                        ),
                    ),
                )


# ============================================================
# VIDEO CONFIG
# ============================================================

VIDEO_WIDTH = 960
VIDEO_HEIGHT = 720
VIDEO_FPS = 30.0


# ============================================================
# COLORS
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
GRAY = (160, 160, 160)


# ============================================================
# FONT
# ============================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# TEXT
# ============================================================

def put_text(
    image,
    text,
    x,
    y,
    scale=0.55,
    color=WHITE,
    thickness=1,
):

    cv2.putText(
        image,
        str(text),
        (x + 2, y + 2),
        FONT,
        scale,
        BLACK,
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        str(text),
        (x, y),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# SAFE DISTANCE
# ============================================================

def get_distance(env):

    try:

        return float(
            env._get_target_distance()
        )

    except Exception:

        if (
            env.target_body_id is not None
        ):

            drone_pos = env.data.xpos[
                env.drone_body_id
            ]

            target_pos = env.data.xpos[
                env.target_body_id
            ]

            return float(
                np.linalg.norm(
                    target_pos
                    - drone_pos
                )
            )

        return 0.0


# ============================================================
# TARGET SPEED
# ============================================================

def get_target_speed(env):

    velocity = getattr(
        env,
        "target_velocity",
        None,
    )

    if velocity is None:

        return 0.0

    return float(
        np.linalg.norm(
            velocity
        )
    )


# ============================================================
# TARGET POSITION
# ============================================================

def get_target_position(env):

    if (
        getattr(
            env,
            "target_is_mocap",
            False,
        )
        and getattr(
            env,
            "target_mocap_id",
            -1,
        ) >= 0
    ):

        return np.array(
            env.data.mocap_pos[
                env.target_mocap_id
            ],
            dtype=np.float32,
        )

    if env.target_body_id is not None:

        return np.array(
            env.data.xpos[
                env.target_body_id
            ],
            dtype=np.float32,
        )

    return np.zeros(
        3,
        dtype=np.float32,
    )


# ============================================================
# DRONE POSITION
# ============================================================

def get_drone_position(env):

    return np.array(
        env.data.xpos[
            env.drone_body_id
        ],
        dtype=np.float32,
    )


# ============================================================
# PANEL
# ============================================================

def draw_panel(
    frame,
    env,
    action,
    reward,
    step,
    episode,
):

    h, w = frame.shape[:2]

    panel_width = 350

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width, h),
        BLACK,
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
    )

    det = VisualInfo(env)

    distance = get_distance(env)

    target_speed = get_target_speed(env)

    altitude = float(
        env.data.xpos[
            env.drone_body_id
        ][2]
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    put_text(
        frame,
        "VISUAL FOLLOW V16.3",
        15,
        30,
        0.70,
        CYAN,
        2,
    )

    put_text(
        frame,
        f"EPISODE {episode}",
        15,
        58,
    )

    put_text(
        frame,
        f"STEP {step}",
        15,
        82,
    )

    # --------------------------------------------------------
    # AGENT INPUT
    # --------------------------------------------------------

    y = 120

    put_text(
        frame,
        "=== AGENT INPUT ===",
        15,
        y,
        0.55,
        YELLOW,
    )

    y += 27

    put_text(
        frame,
        f"IMG: {CFG.camera_height}x{CFG.camera_width}x3",
        15,
        y,
    )

    y += 25

    put_text(
        frame,
        "STATE: 11 values",
        15,
        y,
    )

    # --------------------------------------------------------
    # VISUAL
    # --------------------------------------------------------

    y += 40

    put_text(
        frame,
        "=== VISUAL ===",
        15,
        y,
        0.55,
        YELLOW,
    )

    y += 27

    if det.visible:

        visible_color = GREEN
        visible_text = "YES"

    else:

        visible_color = RED
        visible_text = "NO"

    put_text(
        frame,
        f"VISIBLE    : {visible_text}",
        15,
        y,
        color=visible_color,
    )

    y += 25

    put_text(
        frame,
        f"DX         : {det.dx:+.3f}",
        15,
        y,
    )

    y += 25

    put_text(
        frame,
        f"DY         : {det.dy:+.3f}",
        15,
        y,
    )

    y += 25

    put_text(
        frame,
        f"SIZE       : {det.size:.4f}",
        15,
        y,
    )

    y += 25

    put_text(
        frame,
        f"CONF       : {det.confidence:.3f}",
        15,
        y,
    )

    # --------------------------------------------------------
    # FC
    # --------------------------------------------------------

    y += 40

    put_text(
        frame,
        "=== FLIGHT CONTROLLER ===",
        15,
        y,
        0.55,
        YELLOW,
    )

    y += 27

    put_text(
        frame,
        f"ALTITUDE   : {altitude:.2f} m",
        15,
        y,
    )

    y += 25

    airborne = altitude > 0.5

    put_text(
        frame,
        f"AIRBORNE   : {airborne}",
        15,
        y,
        color=GREEN if airborne else RED,
    )

    y += 25

    takeoff = bool(
        getattr(
            env,
            "takeoff_completed",
            False,
        )
    )

    put_text(
        frame,
        f"TAKEOFF    : {takeoff}",
        15,
        y,
        color=GREEN if takeoff else RED,
    )

    y += 25

    phase = getattr(
        env,
        "phase",
        -1,
    )

    put_text(
        frame,
        f"PHASE      : {phase}",
        15,
        y,
    )

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    y += 40

    put_text(
        frame,
        "=== AGENT ACTION ===",
        15,
        y,
        0.55,
        YELLOW,
    )

    y += 27

    action = np.asarray(
        action,
        dtype=np.float32,
    ).flatten()

    if len(action) >= 4:

        roll = float(action[0])
        pitch = float(action[1])
        yaw = float(action[2])
        thrust = float(action[3])

    else:

        roll = 0.0
        pitch = 0.0
        yaw = 0.0
        thrust = 0.0

    put_text(
        frame,
        f"ROLL       : {roll:+.3f}",
        15,
        y,
    )

    y += 25

    put_text(
        frame,
        f"PITCH      : {pitch:+.3f}",
        15,
        y,
    )

    y += 25

    put_text(
        frame,
        f"YAW        : {yaw:+.3f}",
        15,
        y,
    )

    y += 25

    put_text(
        frame,
        f"THRUST     : {thrust:+.3f}",
        15,
        y,
    )

    # ========================================================
    # DEBUG PANEL
    # ========================================================

    debug_x = w - 330

    y = 30

    put_text(
        frame,
        "SIMULATOR DEBUG",
        debug_x,
        y,
        0.65,
        GRAY,
        2,
    )

    y += 40

    put_text(
        frame,
        f"TRUE DIST  : {distance:.2f} m",
        debug_x,
        y,
        color=RED,
    )

    y += 28

    put_text(
        frame,
        f"TARGET SPD : {target_speed:.2f} m/s",
        debug_x,
        y,
        color=RED,
    )

    y += 28

    put_text(
        frame,
        f"REWARD     : {reward:+.3f}",
        debug_x,
        y,
    )

    y += 28

    put_text(
        frame,
        f"PHASE      : {phase}",
        debug_x,
        y,
    )

    # --------------------------------------------------------
    # TARGET POSITION
    # --------------------------------------------------------

    y += 40

    put_text(
        frame,
        "TARGET POSITION",
        debug_x,
        y,
        0.55,
        GRAY,
    )

    target_pos = get_target_position(env)

    y += 25

    put_text(
        frame,
        f"X: {target_pos[0]:+.1f} m",
        debug_x,
        y,
        0.45,
        GRAY,
    )

    y += 22

    put_text(
        frame,
        f"Y: {target_pos[1]:+.1f} m",
        debug_x,
        y,
        0.45,
        GRAY,
    )

    y += 22

    put_text(
        frame,
        f"Z: {target_pos[2]:+.1f} m",
        debug_x,
        y,
        0.45,
        GRAY,
    )

    # --------------------------------------------------------
    # DRONE POSITION
    # --------------------------------------------------------

    y += 40

    put_text(
        frame,
        "DRONE POSITION",
        debug_x,
        y,
        0.55,
        GRAY,
    )

    drone_pos = get_drone_position(env)

    y += 25

    put_text(
        frame,
        f"X: {drone_pos[0]:+.1f} m",
        debug_x,
        y,
        0.45,
        GRAY,
    )

    y += 22

    put_text(
        frame,
        f"Y: {drone_pos[1]:+.1f} m",
        debug_x,
        y,
        0.45,
        GRAY,
    )

    y += 22

    put_text(
        frame,
        f"Z: {drone_pos[2]:+.1f} m",
        debug_x,
        y,
        0.45,
        GRAY,
    )

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    put_text(
        frame,
        "DEBUG DATA IS NOT OBSERVATION",
        debug_x,
        h - 25,
        0.42,
        GRAY,
    )

    return frame


# ============================================================
# DRAW TARGET
# ============================================================

def draw_detection(
    frame,
    env,
):

    det = VisualInfo(env)

    h, w = frame.shape[:2]

    cx = w // 2
    cy = h // 2

    # --------------------------------------------------------
    # CAMERA CENTER
    # --------------------------------------------------------

    cv2.circle(
        frame,
        (cx, cy),
        6,
        RED,
        2,
    )

    if not det.visible:

        put_text(
            frame,
            "TARGET LOST",
            cx - 90,
            cy - 20,
            0.6,
            RED,
            2,
        )

        return frame

    # --------------------------------------------------------
    # TARGET POSITION
    # --------------------------------------------------------

    tx = int(
        cx
        + det.dx * cx
    )

    ty = int(
        cy
        + det.dy * cy
    )

    tx = max(
        0,
        min(w - 1, tx),
    )

    ty = max(
        0,
        min(h - 1, ty),
    )

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    if det.confidence >= 0.7:

        color = GREEN

    elif det.confidence >= 0.4:

        color = YELLOW

    else:

        color = RED

    # --------------------------------------------------------
    # CROSSHAIR
    # --------------------------------------------------------

    size = 18

    cv2.line(
        frame,
        (tx - size, ty),
        (tx + size, ty),
        color,
        2,
    )

    cv2.line(
        frame,
        (tx, ty - size),
        (tx, ty + size),
        color,
        2,
    )

    # --------------------------------------------------------
    # BOX
    # --------------------------------------------------------

    radius = max(
        10,
        int(
            det.size
            * w
            * 0.5
        ),
    )

    radius = min(
        radius,
        120,
    )

    x1 = max(
        0,
        tx - radius,
    )

    y1 = max(
        0,
        ty - radius,
    )

    x2 = min(
        w - 1,
        tx + radius,
    )

    y2 = min(
        h - 1,
        ty + radius,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    put_text(
        frame,
        (
            f"TARGET "
            f"dx={det.dx:+.2f} "
            f"dy={det.dy:+.2f}"
        ),
        min(
            tx + radius + 8,
            w - 300,
        ),
        max(
            25,
            ty,
        ),
        0.45,
        color,
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    deterministic = not args.stochastic

    print()
    print("=" * 70)
    print("VISUAL FOLLOW SAC V16.3 — TEST")
    print("=" * 70)

    print(
        f"Model       : {args.model}"
    )

    print(
        f"XML         : {args.xml}"
    )

    print(
        f"Episodes    : {args.episodes}"
    )

    print(
        f"Max steps   : {args.max_steps}"
    )

    print(
        f"Video       : {not args.no_video}"
    )

    print(
        f"Window      : {not args.no_window}"
    )

    print(
        f"Phase       : {args.phase}"
    )

    print(
        f"Policy      : "
        f"{'deterministic' if deterministic else 'stochastic'}"
    )

    print("=" * 70)
    print()

    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not os.path.exists(
        args.model
    ):

        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    # ========================================================
    # ENV
    # ========================================================

    print(
        "[ENV] Creating VisualFollowEnv..."
    )

    env = VisualFollowEnv(
        xml_path=args.xml,
        render_mode="rgb_array",
    )

    # --------------------------------------------------------
    # CURRICULUM PHASE
    # --------------------------------------------------------

    if hasattr(
        env,
        "set_phase",
    ):

        env.set_phase(
            args.phase
        )

    else:

        print(
            "[WARN] Environment has no set_phase()"
        )

    print()

    print(
        "[ENV] Observation space:"
    )

    print(
        env.observation_space
    )

    print()

    print(
        "[ENV] Action space:"
    )

    print(
        env.action_space
    )

    print()

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "[MODEL] Loading V16.3..."
    )

    model = SAC.load(
        args.model,
        env=env,
        device="cpu",
    )

    print(
        "[MODEL] Loaded."
    )

    print()

    # ========================================================
    # OUTPUT
    # ========================================================

    os.makedirs(
        args.output,
        exist_ok=True,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    episode_results = []

    # ========================================================
    # EPISODES
    # ========================================================

    for episode in range(
        1,
        args.episodes + 1,
    ):

        print()
        print("=" * 70)
        print(
            f"START EPISODE {episode}"
        )
        print("=" * 70)

        seed = (
            args.seed
            + episode
        )

        obs, reset_info = env.reset(
            seed=seed
        )

        total_reward = 0.0

        start_time = time.time()

        video_writer = None
        video_path = None

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        visible_steps = 0

        dx_sum = 0.0
        dy_sum = 0.0

        distance_sum = 0.0

        min_distance = float(
            "inf"
        )

        max_altitude = -float(
            "inf"
        )

        final_info = {}

        final_reason = "unknown"

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        if not args.no_video:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            video_path = os.path.join(
                args.output,
                (
                    f"visual_follow_"
                    f"ep{episode}_"
                    f"{timestamp}.mp4"
                ),
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                VIDEO_FPS,
                (
                    VIDEO_WIDTH,
                    VIDEO_HEIGHT,
                ),
            )

            if not video_writer.isOpened():

                print(
                    "[WARN] VideoWriter failed"
                )

                video_writer = None

        # ====================================================
        # EPISODE LOOP
        # ====================================================

        for step in range(
            args.max_steps
        ):

            # ------------------------------------------------
            # POLICY
            # ------------------------------------------------

            action, _ = model.predict(
                obs,
                deterministic=deterministic,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            ).flatten()

            # ------------------------------------------------
            # ENV STEP
            # ------------------------------------------------

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            reward = float(
                reward
            )

            total_reward += reward

            final_info = info

            # ------------------------------------------------
            # DEBUG STATISTICS
            # ------------------------------------------------

            det = VisualInfo(env)

            distance = get_distance(
                env
            )

            altitude = float(
                env.data.xpos[
                    env.drone_body_id
                ][2]
            )

            distance_sum += distance

            min_distance = min(
                min_distance,
                distance,
            )

            max_altitude = max(
                max_altitude,
                altitude,
            )

            if det.visible:

                visible_steps += 1

                dx_sum += abs(
                    det.dx
                )

                dy_sum += abs(
                    det.dy
                )

            # ------------------------------------------------
            # CAMERA
            # ------------------------------------------------

            frame = env.render()

            frame = np.asarray(
                frame,
                dtype=np.uint8,
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR,
            )

            # ------------------------------------------------
            # UPSCALE
            # ------------------------------------------------

            frame = cv2.resize(
                frame,
                (
                    VIDEO_WIDTH,
                    VIDEO_HEIGHT,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            # ------------------------------------------------
            # TARGET
            # ------------------------------------------------

            frame = draw_detection(
                frame,
                env,
            )

            # ------------------------------------------------
            # PANEL
            # ------------------------------------------------

            frame = draw_panel(
                frame,
                env,
                action,
                reward,
                step,
                episode,
            )

            # ------------------------------------------------
            # VIDEO
            # ------------------------------------------------

            if video_writer is not None:

                video_writer.write(
                    frame
                )

            # ------------------------------------------------
            # WINDOW
            # ------------------------------------------------

            if not args.no_window:

                cv2.imshow(
                    "VISUAL FOLLOW V16.3",
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    print(
                        "\nUser stopped."
                    )

                    final_reason = (
                        "user_interrupt"
                    )

                    terminated = True

                    break

                if key == 27:

                    print(
                        "\nESC pressed."
                    )

                    final_reason = (
                        "user_interrupt"
                    )

                    terminated = True

                    break

            # ------------------------------------------------
            # CONSOLE
            # ------------------------------------------------

            if (
                step % 10 == 0
                or terminated
                or truncated
            ):

                print(
                    f"STEP {step:4d} | "
                    f"ALT {altitude:5.2f} | "
                    f"DIST {distance:6.2f} | "
                    f"VIS {str(det.visible):5s} | "
                    f"DX {det.dx:+.2f} | "
                    f"DY {det.dy:+.2f} | "
                    f"A ["
                    f"{action[0]:+.2f} "
                    f"{action[1]:+.2f} "
                    f"{action[2]:+.2f} "
                    f"{action[3]:+.2f}"
                    f"] | "
                    f"R {reward:+.3f}"
                )

            # ------------------------------------------------
            # DONE
            # ------------------------------------------------

            if (
                terminated
                or truncated
            ):

                if (
                    "reason"
                    in info
                ):

                    final_reason = str(
                        info["reason"]
                    )

                elif (
                    "episode_end_reason"
                    in info
                ):

                    final_reason = str(
                        info[
                            "episode_end_reason"
                        ]
                    )

                elif truncated:

                    final_reason = (
                        "timeout"
                    )

                break

        # ====================================================
        # CLEAN VIDEO
        # ====================================================

        if video_writer is not None:

            video_writer.release()

        # ====================================================
        # FINAL VALUES
        # ====================================================

        steps_done = step + 1

        final_distance = get_distance(
            env
        )

        final_altitude = float(
            env.data.xpos[
                env.drone_body_id
            ][2]
        )

        target_speed = get_target_speed(
            env
        )

        visible_ratio = (
            visible_steps
            / max(
                steps_done,
                1,
            )
        )

        average_distance = (
            distance_sum
            / max(
                steps_done,
                1,
            )
        )

        if visible_steps > 0:

            average_abs_dx = (
                dx_sum
                / visible_steps
            )

            average_abs_dy = (
                dy_sum
                / visible_steps
            )

        else:

            average_abs_dx = 0.0
            average_abs_dy = 0.0

        airborne = (
            final_altitude > 0.5
        )

        takeoff_completed = bool(
            getattr(
                env,
                "takeoff_completed",
                False,
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        # ====================================================
        # RESULT
        # ====================================================

        result = {

            "episode":
                episode,

            "steps":
                steps_done,

            "reward":
                total_reward,

            "reason":
                final_reason,

            "final_distance":
                final_distance,

            "average_distance":
                average_distance,

            "minimum_distance":
                min_distance,

            "final_altitude":
                final_altitude,

            "max_altitude":
                max_altitude,

            "target_speed":
                target_speed,

            "visible_steps":
                visible_steps,

            "visible_ratio":
                visible_ratio,

            "average_abs_dx":
                average_abs_dx,

            "average_abs_dy":
                average_abs_dy,

            "airborne":
                airborne,

            "takeoff_completed":
                takeoff_completed,

            "elapsed":
                elapsed,

            "video":
                video_path,
        }

        episode_results.append(
            result
        )

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print()
        print("=" * 70)
        print(
            f"EPISODE {episode} FINISHED"
        )
        print("=" * 70)

        print(
            f"Steps              : "
            f"{steps_done}"
        )

        print(
            f"Reward             : "
            f"{total_reward:+.3f}"
        )

        print(
            f"Final distance     : "
            f"{final_distance:.3f} m"
        )

        print(
            f"Average distance   : "
            f"{average_distance:.3f} m"
        )

        print(
            f"Minimum distance   : "
            f"{min_distance:.3f} m"
        )

        print(
            f"Final altitude     : "
            f"{final_altitude:.3f} m"
        )

        print(
            f"Maximum altitude   : "
            f"{max_altitude:.3f} m"
        )

        print(
            f"Target speed       : "
            f"{target_speed:.3f} m/s"
        )

        print(
            f"Visible steps      : "
            f"{visible_steps}"
        )

        print(
            f"Visible ratio      : "
            f"{visible_ratio * 100:.1f}%"
        )

        print(
            f"Average |DX|       : "
            f"{average_abs_dx:.3f}"
        )

        print(
            f"Average |DY|       : "
            f"{average_abs_dy:.3f}"
        )

        print(
            f"Airborne           : "
            f"{airborne}"
        )

        print(
            f"Takeoff complete   : "
            f"{takeoff_completed}"
        )

        print(
            f"Reason             : "
            f"{final_reason}"
        )

        if video_path:

            print(
                f"Video              : "
                f"{video_path}"
            )

        print("=" * 70)

    # ========================================================
    # CLEANUP
    # ========================================================

    if not args.no_window:

        cv2.destroyAllWindows()

    env.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("VISUAL FOLLOW V16.3 — TEST SUMMARY")
    print("=" * 90)

    for result in episode_results:

        print(
            f"EP {result['episode']:2d} | "
            f"steps={result['steps']:4d} | "
            f"reward={result['reward']:+8.2f} | "
            f"dist={result['final_distance']:6.2f} | "
            f"avg_dist={result['average_distance']:6.2f} | "
            f"min_dist={result['minimum_distance']:6.2f} | "
            f"vis={result['visible_ratio'] * 100:5.1f}% | "
            f"dx={result['average_abs_dx']:.3f} | "
            f"dy={result['average_abs_dy']:.3f} | "
            f"alt={result['final_altitude']:5.2f} | "
            f"reason={result['reason']}"
        )

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    if episode_results:

        rewards = [
            r["reward"]
            for r in episode_results
        ]

        distances = [
            r["final_distance"]
            for r in episode_results
        ]

        visible_ratios = [
            r["visible_ratio"]
            for r in episode_results
        ]

        print()
        print("=" * 90)
        print("GLOBAL STATISTICS")
        print("=" * 90)

        print(
            f"Mean reward        : "
            f"{np.mean(rewards):+.3f}"
        )

        print(
            f"Std reward         : "
            f"{np.std(rewards):.3f}"
        )

        print(
            f"Mean final dist    : "
            f"{np.mean(distances):.3f} m"
        )

        print(
            f"Mean visibility    : "
            f"{np.mean(visible_ratios) * 100:.1f}%"
        )

        print("=" * 90)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()