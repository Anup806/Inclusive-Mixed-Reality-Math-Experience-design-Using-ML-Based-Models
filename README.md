# Inclusive Mixed Reality Math Experience Design Using ML-Based Models

> An adaptive, accessible mathematics learning platform that combines gesture recognition, real-time computer vision, and Bayesian Knowledge Tracing to deliver personalized math education for every learner.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev/)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation Instructions](#installation-instructions)
- [Usage Guide](#usage-guide)
- [Folder Structure](#folder-structure)
- [Example Workflow](#example-workflow)
- [Future Improvements](#future-improvements)
- [Contribution Guidelines](#contribution-guidelines)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Inclusive Mixed Reality Math Experience Design Using ML-Based Models** is an innovative educational application that revolutionizes mathematics learning through mixed reality interaction. The system integrates:

- **ML-based gesture recognition** via MediaPipe to detect hand poses and pinch gestures in real time
- **Bayesian Knowledge Tracing (BKT)** to model each student's mastery level and dynamically adapt question difficulty
- **PC-BKT (K-Means clustering)** to classify students into `BASIC`, `INTERMEDIATE`, or `ADVANCED` tiers
- **Audio feedback** with counting sounds and result announcements to support diverse learners
- **Session analytics** logged to CSV for further research and model improvement

This project demonstrates how machine learning models can create inclusive, adaptive learning experiences that respond to individual student needs in real time.

### Two Gameplay Modes

| Mode | Description |
|------|-------------|
| **Without Model** (`Without Model.py`) | Gesture-driven drag-and-drop math game without adaptive difficulty |
| **With Model** (`With Model.py`) | Full adaptive pipeline — BKT engine + PC-BKT K-Means classification + session persistence |

---

## Problem Statement

Traditional mathematics education tools often fail to accommodate diverse learner needs. Students who are blind, deaf, or have learning differences are frequently excluded from standard interactive software. Additionally, most educational applications use a one-size-fits-all approach that doesn't adapt to individual learning pace or style.

This project addresses these gaps by building an inclusive, gesture-driven mixed reality math environment that:

1. **Adapts** problem difficulty in real time based on a student's demonstrated knowledge state
2. **Provides** multimodal feedback (visual + audio) to serve multiple learner types
3. **Records** and persists per-student learning data across sessions for longitudinal tracking
4. **Accessible** without specialized hardware — a standard webcam is sufficient

---

## Objectives

- **O1** – Design and implement a gesture-based mathematics game using hand-tracking (MediaPipe + OpenCV)
- **O2** – Integrate Bayesian Knowledge Tracing (BKT) to estimate per-student mastery probability (`p_know`)
- **O3** – Apply K-Means clustering (PC-BKT) on multi-feature vectors to classify students into difficulty tiers
- **O4** – Persist student progress across sessions and adapt question pools accordingly
- **O5** – Provide rich multimodal feedback (visual animations, audio counts, result sounds) for inclusive learning
- **O6** – Log every interaction and session summary to CSV for post-hoc analysis and model retraining

---

## Features

- 🖐 **Real-time hand gesture detection** — pinch to grab apples, release over basket to answer
- 🎯 **Two game modes** — Counting (place N apples in basket) and Addition (solve A+B by placing apples)
- 🧠 **Adaptive difficulty** — BKT updates `p_know` after every answer; K-Means model classifies level
- 📊 **Session 1 Assessment** — 15 fixed questions to cold-start BKT before switching to adaptive mode
- 💾 **Cross-session persistence** — student levels and `p_know` saved to `student_levels.json`
- 🔊 **Audio feedback** — counting sounds (1–10), cheer on correct answers
- 📁 **CSV analytics** — per-interaction and per-session CSV logs in `game_data/`
- 🎨 **Fullscreen AR overlay** — live webcam feed rendered under the game canvas
- ♿ **Accessibility-first design** — large UI elements, audio cues, color-contrast feedback

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Student Input                         │
│              (Webcam — pinch gesture / drag apples)          │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│               Computer Vision Layer (OpenCV)                 │
│  • Capture frame  →  BGR→RGB conversion  →  resize/flip      │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│           Gesture Recognition (MediaPipe Hands)              │
│  • 21-landmark hand skeleton                                 │
│  • Thumb-index pinch distance  →  grab / release event       │
│  • Finger-tip (x, y) position  →  drag coordinates          │
└───────────────┬─────────────────────────┬────────────────────┘
                │                         │
                ▼                         ▼
┌──────────────────────┐   ┌─────────────────────────────────┐
│   Game Logic Layer   │   │      ML Adaptive Engine         │
│  (Pygame)            │   │                                 │
│  • Counting mode     │   │  BKT Engine                     │
│  • Addition mode     │   │  • p_know update (Bayes rule)   │
│  • Score tracking    │   │  • slip / guess / learn params  │
│  • Apple/basket UI   │   │                                 │
│  • Audio feedback    │   │  PC-BKT (K-Means, k=3)          │
└──────────┬───────────┘   │  • 5-feature vector             │
           │               │  • cluster → BASIC/INTER/ADV    │
           │               └────────────┬────────────────────┘
           │                            │
           └──────────────┬─────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  Persistence & Analytics                     │
│  • student_levels.json  — per-student level & p_know        │
│  • interactions.csv     — every Q&A event                   │
│  • sessions.csv         — session-level summary             │
└──────────────────────────────────────────────────────────────┘
```

### BKT Update Rule

After each answer, the engine applies the standard BKT Bayes update followed by a learning transition:

```
P(know | evidence) = P(know) × P(evidence | know)
                     ─────────────────────────────
                          P(evidence)

P(know_new) = P(know | evidence) × (1 − P_forget)
            + (1 − P(know | evidence)) × P_learn
```

Default parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `P_INIT`  | 0.30 | Prior probability of mastery |
| `P_LEARN` | 0.20 | Probability of learning on each attempt |
| `P_FORGET`| 0.05 | Probability of forgetting after mastery |
| `P_SLIP`  | 0.10 | Probability of wrong answer despite mastery |
| `P_GUESS` | 0.25 | Probability of correct answer without mastery |

---

## Technology Stack

| Component | Library / Tool | Version |
|-----------|---------------|---------|
| Game engine / UI | [Pygame](https://www.pygame.org/) | ≥ 2.x |
| Computer vision | [OpenCV](https://opencv.org/) | ≥ 4.x |
| Gesture recognition | [MediaPipe](https://mediapipe.dev/) | ≥ 0.10 |
| Numerical computing | [NumPy](https://numpy.org/) | ≥ 1.x |
| ML model (K-Means) | [scikit-learn](https://scikit-learn.org/) | ≥ 1.x |
| Deep learning (optional) | [TensorFlow / Keras](https://www.tensorflow.org/) | ≥ 2.x |
| Model persistence | Python `pickle` | stdlib |
| Data logging | Python `csv` / `json` | stdlib |
| Notebook analysis | [Jupyter](https://jupyter.org/) | any |
| Language | Python | ≥ 3.9 |

---

## Installation Instructions

### Prerequisites

- Python **3.9+**
- A working **webcam**
- OS: Windows 10/11, macOS 12+, or Ubuntu 20.04+

### Step 1: Clone the repository

```bash
git clone https://github.com/Anup806/Inclusive-Mixed-Reality-Math-Experience-design-Using-ML-Based-Models.git
cd Inclusive-Mixed-Reality-Math-Experience-design-Using-ML-Based-Models
```

### Step 2: Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install pygame opencv-python mediapipe numpy scikit-learn
```

To enable the optional Deep Knowledge Tracing (DKT) features:

```bash
pip install tensorflow
```

### Step 4: Verify asset files

Ensure the following files exist in the project root:

```
apple.png
basket0.png  …  basket9.png
1.MP3  …  10.MP3
Cheer.MP3
pc_bkt_model (1).pkl
```

> **Note:** The pre-trained K-Means model (`pc_bkt_model (1).pkl`) is included in the repository. If it is missing, the game falls back to threshold-based difficulty classification automatically.

---

## Usage Guide

### Running the game **without** adaptive difficulty

```bash
python "Without Model.py"
```

### Running the game **with** the BKT adaptive engine

```bash
python "With Model.py"
```

### Startup flow

1. **Enter student details** — name, age, grade (displayed on screen; type and press **Enter**)
2. **Select game mode** — `COUNTING` or `ADDITION` (shown as on-screen buttons, confirmed with a pinch gesture)
3. **Play** — a target number is shown; pinch-grab apples and drag them to the basket to answer
4. **Results** — each answer triggers visual + audio feedback and updates the BKT model
5. **Session end** — press **ESC** or close the window; session summary is saved to CSV

### Controls

| Action | How to perform |
|--------|---------------|
| Grab apple | Pinch (thumb + index finger close together) near an apple |
| Release apple | Open pinch gesture over the basket |
| Submit answer | Drag apple into basket zone — submission is automatic |
| Exit game | Press **ESC** |

### Analyzing data

Open the Jupyter notebook to explore logged interactions and retrain the PC-BKT model:

```bash
jupyter notebook pc_bkt_after_realdata.ipynb
```

---

## Folder Structure

```
Inclusive-Mixed-Reality-Math-Experience-design-Using-ML-Based-Models/
│
├── With Model.py              # Adaptive game (BKT + PC-BKT K-Means)
├── Without Model.py           # Base game (gesture-only, no ML adaptation)
├── pc_bkt_after_realdata.ipynb# Notebook: PC-BKT model training & analysis
├── pc_bkt_model (1).pkl       # Pre-trained K-Means clustering model
├── README.md                  # This file
│
├── apple.png                  # Apple sprite asset
├── basket0.png … basket9.png  # Basket sprites (0–9 apples)
├── 1.MP3 … 10.MP3             # Counting audio cues
├── Cheer.MP3                  # Correct-answer celebration sound
│
└── game_data/                 # Auto-created at runtime
    ├── interactions.csv       # Per-answer log (student, problem, RT, correct, p_know, …)
    ├── sessions.csv           # Per-session summary (accuracy, final_p_know, level, …)
    └── student_levels.json    # Persistent student level & p_know across sessions
```

---

## Example Workflow

### Session 1 (new student "Alice")

```
1. Alice launches "With Model.py"
2. Enters name: Alice  |  age: 14  |  grade: 9
3. Selects game mode: COUNTING
4. Assessment phase begins (15 questions using numbers 1, 2, 3)
5. Alice answers each question by dragging apples into the basket
6. After 15 questions, BKT computes p_know and PC-BKT classifies Alice as INTERMEDIATE
7. Session summary written to sessions.csv; level persisted to student_levels.json
```

### Session 2 (returning student "Alice")

```
1. Alice relaunches the game
2. student_levels.json loads Alice's previous level: INTERMEDIATE
3. Questions drawn from the INTERMEDIATE pool (numbers 4, 5, 6)
4. BKT continues updating p_know; every 5 questions the level is re-evaluated
5. If p_know rises sufficiently, Alice is promoted to ADVANCED (numbers 7–10)
```

### Sample `interactions.csv` entry

```csv
student_name,age,student_grade,game_mode,timestamp,math_problem,user_answer,correct_answer,reaction_time_s,correct,score,session_id,total_screen_time,p_know,difficulty_level
Alice,14,9,COUNTING,2025-12-26 10:20:54.932,3 apples,3,3,8.412,1,10,20251226_102016_6158,38.70,0.5320,INTERMEDIATE
```

---

## Future Improvements

- **Speech interaction** — integrate Vosk or Whisper for voice-command input, enabling fully hands-free gameplay for motor-impaired users
- **Sign language recognition** — extend MediaPipe gesture vocabulary to recognize ASL/BSL number signs
- **Unity / HoloLens port** — migrate the MR overlay to a full 3D Mixed Reality environment using Microsoft MRTK
- **DKT (Deep Knowledge Tracing)** — replace the BKT engine with an LSTM-based DKT model for richer temporal modeling; TensorFlow hooks are already in place
- **Teacher dashboard** — a web interface to visualize class-wide learning analytics from the CSV logs
- **Automated model retraining** — scheduled pipeline to retrain the PC-BKT K-Means model as new interaction data accumulates
- **Multilingual audio** — localized counting sounds and prompts
- **Mobile / tablet support** — touch-gesture fallback when no webcam is available

---

## Contribution Guidelines

Contributions are welcome! Please follow the steps below:

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** — keep commits focused and descriptive.
3. **Test** your changes with both `Without Model.py` and `With Model.py`.
4. **Open a Pull Request** against `main` with a clear description of the change and the problem it solves.

### Code style

- Follow [PEP 8](https://pep8.org/) for Python code.
- Document new functions/classes with docstrings.
- Do not commit large binary assets (images, audio, model files) unless they are required for the feature.

### Reporting issues

Please open a GitHub Issue and include:
- Python version and OS
- Full traceback or error message
- Steps to reproduce

---

## Acknowledgments

This project was developed to advance inclusive education through mixed reality and machine learning. Special thanks to:

- The **MediaPipe** team for providing robust hand tracking capabilities
- The **Pygame** community for excellent documentation and support
- Researchers in **Bayesian Knowledge Tracing** and adaptive learning systems
- All contributors and users who help improve this platform

---

**Built with ❤️ for inclusive education — because every student deserves to learn.**

---

**Repository:** [Anup806/Inclusive-Mixed-Reality-Math-Experience-design-Using-ML-Based-Models](https://github.com/Anup806/Inclusive-Mixed-Reality-Math-Experience-design-Using-ML-Based-Models)