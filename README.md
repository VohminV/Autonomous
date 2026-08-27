Я создал файл `README.md` в рабочей среде. Однако, поскольку я текстовый ИИ и не могу напрямую отправить файл на ваш компьютер, я предоставляю его содержимое ниже в виде единого блока кода. 

Вы можете нажать кнопку **"Copy"** (Копировать) в правом верхнем углу блока и сохранить это как `README.md` в корне вашего проекта.

```markdown
# 🚁 Autonomous — Visual Follow RL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/physics-MuJoCo-red.svg)](https://mujoco.org/)
[![Gymnasium](https://img.shields.io/badge/RL-Gymnasium-orange.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green.svg)](https://stable-baselines3.readthedocs.io/)
[![SAC](https://img.shields.io/badge/algorithm-SAC-purple.svg)](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Исследовательский проект по обучению автономного квадрокоптера визуальному сопровождению движущегося объекта с использованием Reinforcement Learning, MuJoCo и Stable-Baselines3.

---

## 🎯 Цель проекта

Основная задача проекта — обучить квадрокоптер **визуально сопровождать движущуюся цель**, используя информацию, доступную бортовой системе.

Агент должен:

1. Выполнить взлёт.
2. Стабилизировать полёт.
3. Обнаружить движущуюся цель в камере.
4. Удерживать цель в поле зрения.
5. Следовать за целью.
6. Поддерживать безопасную дистанцию.
7. Реагировать на потерю визуального контакта.
8. Работать в условиях шума, задержек и нестабильной детекции.

Все эксперименты выполняются в физической симуляции MuJoCo.

---

# 🧠 Текущая версия

## V16.3

Текущая архитектура использует:

- MuJoCo
- Gymnasium
- Stable-Baselines3
- SAC — Soft Actor-Critic
- визуальный вход камеры
- state vector
- Virtual Flight Controller
- PID-регуляторы
- curriculum learning
- движущуюся цель
- визуальную модель обнаружения цели

### Observation

Агент получает два типа входных данных:

```text
IMAGE
48 × 48 × 3

STATE
11 values
```

Observation реализован как `Dict` space.

Агент не получает напрямую истинную дистанцию до цели, её мировые координаты или мировую скорость.

---

# 🏗️ Архитектура

```text
                    ┌──────────────────────────┐
                    │        MuJoCo            │
                    │                          │
                    │  ┌───────┐   ┌───────┐  │
                    │  │ Drone │   │ Target│  │
                    │  │  x2   │   │moving │  │
                    │  └───┬───┘   └───┬───┘  │
                    └──────┼───────────┼──────┘
                           │           │
                           ▼           ▼
                    ┌──────────────────────────┐
                    │    Sensor Simulation     │
                    │                          │
                    │ Camera / Visual Target   │
                    │ IMU / Altitude / State   │
                    └────────────┬─────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
          ┌────────────────┐           ┌────────────────┐
          │  Image 48x48   │           │   State (11)   │
          │      RGB       │           │                │
          └───────┬────────┘           └───────┬────────┘
                  │                            │
                  └────────────┬───────────────┘
                               ▼
                    ┌──────────────────────────┐
                    │       SAC Agent          │
                    │                          │
                    │   Stable-Baselines3      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       Action (4D)        │
                    │                          │
                    │ roll / pitch / yaw /     │
                    │ thrust                   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Virtual Flight Controller│
                    │                          │
                    │       PID control        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      4 Motor Outputs     │
                    │                          │
                    │ M1 / M2 / M3 / M4        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                              MuJoCo
```

---

# 👁️ Observation Space

## Camera

Камера агента:

```text
48 × 48 × 3
```

RGB image используется как визуальный источник информации.

---

## State

Текущий state содержит **11 значений**, необходимых для управления и визуального сопровождения.

В зависимости от текущей конфигурации среды state включает визуальные признаки цели и параметры состояния полёта.

Точный состав определяется реализацией `VisualFollowEnv` в `train_rl_agent.py`.

### Важно

Истинные параметры симуляции используются только для reward/debugging.

Например:

```text
true distance
target world position
target world velocity
```

не передаются напрямую SAC policy.

---

# 🎮 Action Space

Агент управляет дроном четырьмя непрерывными действиями:

```text
[roll, pitch, yaw, thrust]
```

Action space:

```text
[-1, +1] × 4
```

Высокоуровневые команды передаются в Virtual Flight Controller, который преобразует их в управление четырьмя моторами.

---

# 🛫 Curriculum Learning

Обучение выполняется по уровням сложности.

Текущая версия использует curriculum level для постепенного усложнения задачи.

Примерная логика:

```text
LEVEL 0
  ↓
взлёт и базовая стабилизация

LEVEL 1
  ↓
удержание высоты

LEVEL 2
  ↓
визуальное обнаружение цели

LEVEL 3
  ↓
визуальное сопровождение движущейся цели

LEVEL 4
  ↓
устойчивое follow-поведение
```

Конкретные параметры curriculum определяются текущей реализацией `VisualFollowEnv`.

Это позволяет SAC сначала изучить базовое управление, а затем переходить к более сложному визуальному поведению.

---

# 🎯 Reward Design

В V16.3 reward переработан для более стабильного обучения.

Основные принципы:

* reward за удержание цели в визуальном центре;
* reward за корректное визуальное сопровождение;
* штраф за потерю цели;
* ограниченный и bounded reward;
* отсутствие постоянного большого visibility tax;
* дополнительные сигналы для стабилизации поведения;
* curriculum-dependent reward shaping.

Главная цель reward:

```text
KEEP TARGET VISIBLE
        +
KEEP TARGET CENTERED
        +
MAINTAIN STABLE FLIGHT
```

При этом чрезмерное сближение с целью рассматривается как нежелательное поведение.

---

# 🧠 SAC

Для обучения используется:

**Soft Actor-Critic (SAC)** из Stable-Baselines3.

Основные свойства:

* continuous action space;
* off-policy learning;
* entropy regularization;
* replay buffer;
* устойчивость к сложным непрерывным задачам управления.

Пример параметров обучения:

```text
learning_rate = 3e-4
```

В процессе обучения контролируются:

```text
ep_rew_mean
ep_len_mean
actor_loss
critic_loss
ent_coef
n_updates
```

---

# 📊 Результат обучения V16.3

Промежуточный запуск:

```text
TRAIN
timesteps: 10,000

Eval reward:
-15.43 ± 47.26

Episode length:
732 ± 136

Best model:
visual_follow_v16_3.zip
```

Обучение продолжается до целевого объёма:

```text
200,000 timesteps
```

Текущий результат следует рассматривать как **промежуточный этап обучения**, а не как финальную оценку качества агента.

---

# 🚀 Быстрый старт

## 1. Клонирование

```bash
git clone https://github.com/<your-username>/Autonomous.git
cd Autonomous
```

## 2. Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

---

# 📦 requirements.txt

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

---

# 🏋️ Обучение

Для текущей версии:

```bash
python train_rl_agent.py --mode train --timesteps 200000
```

Пример запуска:

```text
[TRAIN] timesteps=5000 curriculum_level=0
```

По завершении модель сохраняется:

```text
visual_follow_v16_3.zip
```

---

# 🧪 Тестирование

Для тестирования используется:

```text
test_visual_follow.py
```

Пример:

```bash
python test_visual_follow.py --model visual_follow_v16_3.zip --phase 4
```

По умолчанию:

```text
episodes  = 3
max steps = 1000
phase     = 4
seed      = 1000
```

---

# 🎥 Запись видео

Видео тестовых эпизодов сохраняются в:

```text
test_videos/
```

Пример:

```bash
python test_visual_follow.py \
    --model visual_follow_v16_3.zip \
    --phase 4 \
    --episodes 3
```

Для запуска без записи:

```bash
python test_visual_follow.py \
    --model visual_follow_v16_3.zip \
    --phase 4 \
    --no-video
```

Для запуска без графического окна:

```bash
python test_visual_follow.py \
    --model visual_follow_v16_3.zip \
    --phase 4 \
    --no-window
```

---

# 🔬 Test Debug Overlay

Тестовый скрипт выводит расширенную диагностическую информацию.

На экран выводятся:

```text
VISUAL
    visible
    dx
    dy
    size
    confidence

FLIGHT CONTROLLER
    altitude
    airborne
    takeoff status
    curriculum phase

AGENT ACTION
    roll
    pitch
    yaw
    thrust

SIMULATOR DEBUG
    true distance
    target speed
    target position
    drone position
    reward
```

### Важно

Данные:

```text
TRUE DISTANCE
TARGET POSITION
TARGET SPEED
DRONE POSITION
```

используются **только для диагностики**.

Они не передаются агенту как observation.

---

# 📂 Структура проекта

```text
Autonomous/
│
├── train_rl_agent.py
│   └── VisualFollowEnv + SAC training
│
├── test_visual_follow.py
│   └── Model testing + video + debug overlay
│
├── scene.xml
│   └── MuJoCo scene
│
├── requirements.txt
│
├── README.md
│
├── models/
│   └── trained SAC models
│
├── checkpoints/
│   └── training checkpoints
│
├── rl_logs/
│   └── TensorBoard logs
│
└── test_videos/
    └── test recordings
```

---

# ⚙️ MuJoCo Simulation

Симулятор содержит:

* квадрокоптер;
* физическую модель корпуса;
* четыре двигателя;
* движущуюся цель;
* камеру;
* физическое взаимодействие с окружающей средой.

Дрон моделируется как физический объект MuJoCo.

Управление выполняется через моторные актуаторы.

---

# 🛩️ Virtual Flight Controller

RL policy не управляет моторами напрямую.

Схема:

```text
SAC
 ↓
roll / pitch / yaw / thrust
 ↓
Virtual Flight Controller
 ↓
PID
 ↓
motor 1
motor 2
motor 3
motor 4
 ↓
MuJoCo
```

Такой подход разделяет:

```text
HIGH LEVEL
RL policy

LOW LEVEL
flight controller
```

Это позволяет отдельно исследовать RL-политику и стабилизацию летательного аппарата.

---

# 👁️ Visual Follow

Основная задача агента — не знать положение цели в мировых координатах, а использовать визуальную информацию.

Упрощённо:

```text
Camera
  ↓
Target detection
  ↓
dx / dy / visibility
  ↓
SAC
  ↓
Flight command
```

Если цель смещается относительно центра изображения:

```text
dx < 0  → target left
dx > 0  → target right

dy < 0  → target up
dy > 0  → target down
```

Политика учится самостоятельно выбирать управляющее воздействие.

---

# 🧪 Почему MuJoCo

MuJoCo используется потому, что позволяет:

* моделировать динамику квадрокоптера;
* использовать физические ограничения;
* моделировать моторы и силы;
* получать RGB camera observations;
* воспроизводить различные сценарии движения;
* быстро перезапускать большое количество RL episodes.

---

# 📈 TensorBoard

Для просмотра обучения:

```bash
tensorboard --logdir ./rl_logs
```

После запуска:

```text
http://localhost:6006
```

Основные показатели:

```text
rollout/ep_rew_mean
rollout/ep_len_mean
train/actor_loss
train/critic_loss
train/ent_coef
train/n_updates
```

Особенно важны:

### `ep_rew_mean`

Средняя награда эпизода.

Рост показателя обычно означает улучшение поведения policy.

### `ep_len_mean`

Средняя длина эпизода.

Помогает понять, как часто агент достигает terminal condition.

### `critic_loss`

Показывает стабильность обучения критика.

### `ent_coef`

Коэффициент энтропии SAC.

Он отражает баланс между исследованием пространства действий и эксплуатацией уже найденной политики.

---

# 🔧 Текущие ограничения

Проект находится в активной разработке.

Текущие ограничения:

* визуальная модель ещё не является полноценным real-world YOLO;
* camera domain отличается от реальной FPV-камеры;
* физическая модель является приближённой;
* отсутствует полноценная domain randomization;
* перенос policy непосредственно на реальный аппарат требует дополнительной валидации;
* качество policy зависит от curriculum и reward shaping.

---

# 🛠️ План развития

* [ ] Завершить обучение V16.3 на **200k timesteps**
* [ ] Провести серию тестов на разных seed
* [ ] Оценить среднюю дистанцию до цели
* [ ] Оценить стабильность визуального сопровождения
* [ ] Добавить статистику success rate
* [ ] Расширить domain randomization
* [ ] Добавить CNN encoder для raw image
* [ ] Исследовать CNN + LSTM / recurrent policy
* [ ] Добавить более реалистичную модель камеры
* [ ] Интегрировать реальный YOLO inference
* [ ] Исследовать перенос на embedded hardware
* [ ] Интегрировать реальный flight controller после отдельной валидации

---

# 🔐 Safety / Research Scope

Проект предназначен для:

* исследования Reinforcement Learning;
* моделирования управления БПЛА;
* computer vision;
* визуального tracking;
* simulation-based control;
* разработки и тестирования алгоритмов автономной навигации.

Все эксперименты в текущем проекте выполняются в симуляции.

Целью текущей версии является **визуальное сопровождение движущегося объекта и удержание безопасной дистанции**, а не поражение или уничтожение цели.

---

# 📄 Лицензия

MIT License.

См. [LICENSE](LICENSE).

---

# 🙏 Благодарности

* [MuJoCo](https://mujoco.org/) — физический движок
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — RL framework
* [Gymnasium](https://gymnasium.farama.org/) — RL environment API
* [PyTorch](https://pytorch.org/) — deep learning backend
* [OpenCV](https://opencv.org/) — computer vision

---

# 📌 Project Status

Проект находится в стадии активного исследования и оптимизации RL policy.
```

