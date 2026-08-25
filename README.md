# 🚁 Autonomous — Reinforcement Learning для автономного дрона-перехватчика

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MuJoCo](https://img.shields.io/badge/physics-MuJoCo-red.svg)](https://mujoco.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green.svg)](https://stable-baselines3.readthedocs.io/)

> Физически достоверная среда для обучения **автономного дрона-перехватчика** методами глубокого обучения с подкреплением (RL). Агент управляет квадрокоптером **только через бортовые сенсоры** (камера + IMU + GPS + оптический поток) — без читерских ground-truth координат.

---

## 🎯 Цель проекта

Разработать и обучить нейросетевую политику, которая в полностью автономном режиме:

1. **Взлетает** с земли и набирает крейсерскую высоту
2. **Ориентируется** в пространстве и летит к зоне перехвата
3. **Обнаруживает** маневрирующую цель через эмулятор YOLO (с шумом, задержками и dropout)
4. **Преследует** цель, применяя упреждение через визуальную скорость (`v_dx`, `v_dy`)
5. **Облетает** препятствия через оптический поток (3 луча: forward/left/right)
6. **Перехватывает** цель (дистанция < 2 м)

Всё это — **без внешних систем позиционирования цели**, только на данных, доступных реальному бортовому компьютеру.

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    MuJoCo Physics Engine                    │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │  Drone   │   │    Target    │   │   Obstacles      │   │
│  │ (x2)     │   │  (mocap)     │   │   (buildings)    │   │
│  └────┬─────┘   └──────┬───────┘   └────────┬─────────┘   │
│       │                │                    │              │
└───────┼────────────────┼────────────────────┼──────────────┘
        │                │                    │
        ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Sensor Simulation Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ YOLO Emulator│  │ Optical Flow │  │ IMU / Baro / GPS │  │
│  │ (noise+delay)│  │  (3 rays)    │  │    (ground truth)│  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
└─────────┼─────────────────┼───────────────────┼────────────┘
          │                 │                   │
          └─────────────────┼───────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Observation Vector  │
                 │    (23 features)    │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │   SAC Agent (MLP)   │
                 │  [256, 256] + Tanh  │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │   Action (4D)       │
                 │ roll, pitch, yaw,   │
                 │ thrust              │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Virtual FC (PID)    │
                 │ → Motor commands    │
                 └─────────────────────┘
```

---

## 📦 Observation Space (23 признака)

**Честное визуальное сервирование — никаких ground-truth координат цели.**

| №   | Признак                  | Источник          | Описание |
|-----|--------------------------|-------------------|----------|
| 0-4 | `dx, dy, visible, size, confidence` | YOLO Emulator | Bounding box цели в кадре |
| 5-6 | `v_dx, v_dy`             | Optical flow (target) | Визуальная скорость цели (для упреждения) |
| 7-10| `roll, pitch, sin_yaw, cos_yaw` | IMU | Ориентация дрона |
| 11-12| `altitude, yaw_rate`    | Barometer + IMU | Высота и скорость рыскания |
| 13-15| `last_dx, last_dy, lost_time` | Memory | Память о последнем положении цели |
| 16-17| `locked, lock_counter`  | Lock logic | Статус захвата цели |
| 18-19| `drone_x, drone_y`      | GPS | Позиция дрона |
| 20-22| `flow_fwd, flow_left, flow_right` | Optical flow (obstacles) | Облёт препятствий |

---

## 🎮 Curriculum Learning

Обучение разбито на **6 фаз** с transfer learning между ними:

| Фаза | Шаги | Цель |
|------|------|------|
| `TAKEOFF` | 200k | Набор высоты с земли |
| `HOVER` | 200k | Удержание на крейсерской высоте |
| `SEARCH` | 200k | Сканирование пространства |
| `PURSUIT` | 300k | Преследование с упреждением |
| `INTERCEPT` | 400k | Финальное сближение |
| `MISSION` | 800k | **Сквозная миссия** (взлёт → перехват) |

Каждая фаза имеет **свою статистику `VecNormalize`**, чтобы избежать интерференции распределений.

---

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Клонируем репозиторий
git clone https://github.com/<your-username>/Autonomous.git
cd Autonomous

# Создаём виртуальное окружение
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# Устанавливаем зависимости
pip install -r requirements.txt
```

### `requirements.txt`

```txt
mujoco>=3.1.0
gymnasium>=0.29.0
stable-baselines3>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
torch>=2.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
tensorboard>=2.14.0
```

### Обучение

```bash
# Полное обучение с нуля (все 6 фаз)
python train_rl_agent.py --detector synthetic_yolo --noise-level medium

# Продолжение с определённой фазы (индекс 0-5)
python train_rl_agent.py --start-phase 3

# Обучение с идеальным детектором (для отладки)
python train_rl_agent.py --detector oracle
```

### Тестирование обученной модели

```bash
# Тест фазы PURSUIT с сохранением видео
python test_rl_agent.py --phase pursuit --detector synthetic_yolo

# Тест сквозной миссии без окна
python test_rl_agent.py --phase mission --no-show
```

---

## 📂 Структура проекта

```
Autonomous/
├── drone_intercept_env.py    # Основная среда (Gymnasium)
├── train_rl_agent.py         # Скрипт обучения (SAC + curriculum)
├── test_rl_agent.py          # Визуализация и запись видео
├── scene.xml                 # MuJoCo сцена (дрон, цель, здания)
├── requirements.txt          # Зависимости Python
├── README.md                 # Этот файл
│
├── models/                   # Обученные модели (.zip) и нормализации (.pkl)
├── checkpoints/              # Промежуточные чекпоинты по фазам
├── rl_logs/                  # Логи TensorBoard
└── test_videos/              # Записанные видео тестовых прогонов
```

---

## 🧠 Ключевые инженерные решения

### 1. Визуальное сервирование вместо ground-truth
Агент **не знает** истинные координаты цели. Он видит только `dx, dy` в пикселях и учится упреждать маневры через визуальную скорость `v_dx = (dx_t - dx_{t-1}) / dt`. Это позволяет **бесшовно переносить политику на реальное железо**.

### 2. Оптический поток для облёта препятствий
3 виртуальных луча (forward, left±30°) через `mujoco.mj_ray()` имитируют оптический поток. В реальности этот же интерфейс заменяется на `cv2.calcOpticalFlowPyrLK()`.

### 3. Эмулятор YOLO с реалистичным шумом
- **Latency buffer** — задержка детекции 60 мс
- **False negative rate** — зависит от размера цели
- **Dropout** — случайные потери трека
- **BBox noise** — джиттер координат

### 4. Virtual Flight Controller (PID)
Агент выдаёт **команды высокого уровня** (roll, pitch, yaw_rate, thrust), а низкоуровневый PID-контроллер распределяет их по 4 моторам. Это соответствует реальному стеку управления (PX4/ArduPilot).

### 5. Reward Shaping без локальных оптимумов
- Линейные награды с **плато** (агент не улетает в стратосферу)
- `closing_speed` reward в фазе PURSUIT (мотивирует сближаться)
- Штраф за `visual_speed` (цель не должна убегать из кадра)
- Экспоненциальный бонус на финальных метрах

---

## 📊 Мониторинг обучения

```bash
# Запуск TensorBoard
tensorboard --logdir ./rl_logs

# Открыть в браузере
http://localhost:6006
```

**Ключевые метрики:**
- `rollout/success_rate` — доля успешных эпизодов
- `rollout/ep_rew_mean` — средняя награда
- `train/ent_coef` — энтропия (не должна падать ниже 0.01)
- `train/critic_loss` — стабильность критика (< 1.0)

---

## 🔧 Перенос на реальное железо

Интерфейс среды спроектирован для **бесшовной замены симуляции на реальные сенсоры**:

| Симуляция | Реальность |
|-----------|------------|
| `mujoco.mj_ray()` | `cv2.calcOpticalFlowPyrLK()` |
| `YOLOEmulator` | Real YOLOv8/v11 inference |
| `VirtualFlightController` | PX4/ArduPilot FC |
| `data.xpos`, `data.qvel` | GPS + IMU (MAVLink) |
| `MAVLinkProtocol.encode_attitude_target()` | Реальный UART/UDP пакет |

Класс `MAVLinkProtocol` уже содержит сериализацию `SET_ATTITUDE_TARGET` (#82) для отправки команд на реальный FC.

---

## 🛠️ Возможные улучшения

- [ ] Переход на **CNN + LSTM** для обработки сырых изображений
- [ ] **Domain Randomization** для robust transfer
- [ ] **Multi-agent** сценарий (рой дронов)
- [ ] Интеграция с **ROS 2** для реального деплоя
- [ ] Поддержка **GPU** обучения (удалить `CUDA_VISIBLE_DEVICES=""`)

---

## 📄 Лицензия

MIT License. См. [LICENSE](LICENSE).

---

##  Благодарности

- [MuJoCo](https://mujoco.org/) — физический движок
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — RL алгоритмы
- [Gymnasium](https://gymnasium.farama.org/) — RL API
- [PyTorch](https://pytorch.org/) — нейросетевой бэкенд

---
