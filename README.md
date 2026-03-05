# Inclusive Mixed Reality Math Experience Design Using ML-Based Models

> **Portfolio Project** — An adaptive, accessible mathematics learning platform that combines gesture recognition, real-time computer vision, and Bayesian Knowledge Tracing to deliver personalized math education.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Implementation Details](#implementation-details)
- [Example Workflow](#example-workflow)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

This project demonstrates the integration of **machine learning**, **computer vision**, and **adaptive learning systems** to create an inclusive educational experience. The platform uses real-time gesture recognition to enable touchless interaction with mathematical problems while dynamically adjusting difficulty based on individual student performance.

### Core Innovations

- **ML-based gesture recognition** via MediaPipe for real-time hand pose detection and pinch gestures
- **Bayesian Knowledge Tracing (BKT)** engine to model student mastery levels (`p_know`) and adapt content dynamically
- **PC-BKT with K-Means clustering** to classify students into `BASIC`, `INTERMEDIATE`, or `ADVANCED` difficulty tiers
- **Multimodal feedback system** with visual animations and audio cues to support diverse learners
- **Session persistence** with analytics logged to CSV for longitudinal tracking and model improvement

### Two Implementation Modes

| Mode | Description |
|------|-------------|
| **Without Model** | Gesture-driven drag-and-drop math game demonstrating computer vision capabilities |
| **With Model** | Full adaptive pipeline integrating BKT engine, PC-BKT K-Means classification, and cross-session persistence |

---

## Problem Statement

Traditional mathematics education tools frequently fail to accommodate diverse learner needs. Students with visual, auditory, or learning differences are often excluded from interactive software that relies on mouse/keyboard input and assumes uniform learning paces.

This project addresses these critical gaps by:

1. **Eliminating hardware barriers** — operates with standard webcam hardware (no specialized sensors required)
2. **Providing adaptive difficulty** — real-time knowledge state estimation ensures appropriate challenge levels
3. **Enabling multimodal feedback** — visual + audio output supports multiple learning styles
4. **Tracking longitudinal progress** — persistent student profiles enable long-term assessment and intervention

---

## Objectives

The system was designed to achieve the following technical and educational objectives:

- **O1** — Implement gesture-based interaction using MediaPipe hand tracking and OpenCV computer vision
- **O2** — Integrate Bayesian Knowledge Tracing to estimate per-student mastery probability in real time
- **O3** — Apply K-Means clustering on multi-feature vectors to classify students into appropriate difficulty tiers
- **O4** — Persist student progress across sessions and dynamically adapt question pools
- **O5** — Deliver rich multimodal feedback (visual animations, audio counting, result sounds)
- **O6** — Log every interaction and session summary for post-hoc analysis and model retraining

---

## Key Features

### Computer Vision & Gesture Recognition
- Real-time hand landmark detection (21-point skeleton via MediaPipe)
- Pinch gesture detection (thumb-index distance threshold)
- Drag-and-drop interaction without physical input devices
- Fullscreen AR overlay with live webcam feed

### Adaptive Learning Engine
- **Bayesian Knowledge Tracing (BKT)** with configurable parameters (slip, guess, learn, forget)
- **PC-BKT K-Means classifier** trained on 5-feature vectors (accuracy, p_know, response time, consistency, engagement)
- Dynamic difficulty adjustment every 5 questions
- Session 1 cold-start assessment (15 fixed questions) before adaptive mode

### Accessibility & Inclusivity
- Gesture-based input eliminates need for fine motor control
- Audio feedback for counting (1–10) and results
- High-contrast UI elements
- Large touch targets for basket interaction
- Multimodal output (visual + auditory)

### Data Collection & Persistence
- Per-interaction CSV logging (timestamp, problem, answer, reaction time, correctness, p_know, level)
- Session summary CSV (total accuracy, final p_know, level classification)
- JSON-based student profile persistence across sessions

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

### Bayesian Knowledge Tracing Implementation

The BKT engine applies standard Bayesian updating after each student response:

```
P(know | evidence) = P(know) × P(evidence | know)
                     ─────────────────────────────
                          P(evidence)

P(know_new) = P(know | evidence) × (1 − P_forget)
            + (1 − P(know | evidence)) × P_learn
```

**Model Parameters:**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `P_INIT`  | 0.30 | Prior probability student has mastered skill |
| `P_LEARN` | 0.20 | Probability of acquiring skill on each attempt |
| `P_FORGET`| 0.05 | Probability of skill degradation after mastery |
| `P_SLIP`  | 0.10 | Probability of error despite mastery (careless mistake) |
| `P_GUESS` | 0.25 | Probability of correct answer without mastery (lucky guess) |

---

## Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Game Engine** | Pygame 2.x | Lightweight rendering, real-time event handling, audio mixing |
| **Computer Vision** | OpenCV 4.x | Industry-standard CV library, efficient frame processing |
| **Gesture Recognition** | MediaPipe 0.10+ | Google's production-grade hand tracking model |
| **Numerical Computing** | NumPy | Efficient array operations for BKT calculations |
| **Machine Learning** | scikit-learn | K-Means clustering for PC-BKT student classification |
| **Optional: Deep Learning** | TensorFlow 2.x | Infrastructure for future DKT (Deep Knowledge Tracing) implementation |
| **Data Persistence** | Python stdlib (`pickle`, `json`, `csv`) | Zero-dependency serialization |
| **Development Language** | Python 3.9+ | Rich ML/CV ecosystem, rapid prototyping |

---

## Implementation Details

### Computer Vision Pipeline
1. **Frame Capture** — OpenCV reads webcam feed at 30 FPS
2. **Preprocessing** — BGR→RGB conversion, horizontal flip for mirror effect
3. **Hand Detection** — MediaPipe processes frame, returns 21 3D landmarks per hand
4. **Gesture Classification** — Euclidean distance between thumb tip (landmark 4) and index tip (landmark 8)
   - Distance < threshold → **pinch detected** (grab apple)
   - Distance > threshold → **release detected** (drop apple)

### Adaptive Difficulty System
1. **Cold Start (Session 1)** — 15 assessment questions using numbers 1–3
2. **BKT Initialization** — `p_know = 0.30` (prior)
3. **Update Loop:**
   - Student answers question → BKT updates `p_know` using Bayes rule
   - Every 5 questions → PC-BKT K-Means model classifies student based on:
     - Cumulative accuracy
     - Current `p_know`
     - Average reaction time
     - Response consistency
     - Engagement metrics
4. **Difficulty Assignment:**
   - **BASIC** → numbers 1–3
   - **INTERMEDIATE** → numbers 4–6
   - **ADVANCED** → numbers 7–10

### Data Collection Schema

**`interactions.csv`** (per-question log):
```csv
student_name, age, student_grade, game_mode, timestamp, math_problem, 
user_answer, correct_answer, reaction_time_s, correct, score, session_id, 
total_screen_time, p_know, difficulty_level
```

**`sessions.csv`** (per-session summary):
```csv
session_id, student_name, age, grade, game_mode, total_questions, 
correct_answers, accuracy, avg_reaction_time_s, final_p_know, 
difficulty_level, session_duration_s
```

---

## Example Workflow

### Session 1: New Student Assessment

```
1. Student "Alex" launches application
2. Enters profile: Name=Alex, Age=12, Grade=7
3. Selects game mode: COUNTING
4. Assessment phase (15 questions, numbers 1–3)
   • Alex answers by pinch-grabbing apples and dragging to basket
   • BKT updates p_know after each answer
5. Post-assessment:
   • Final p_know = 0.62
   • PC-BKT classifies Alex as INTERMEDIATE
6. Data persistence:
   • 15 rows written to interactions.csv
   • 1 row written to sessions.csv
   • Alex's profile saved to student_levels.json
```

### Session 2: Returning Student

```
1. Alex relaunches application
2. student_levels.json loads saved state:
   • Previous level: INTERMEDIATE
   • Previous p_know: 0.62
3. Questions drawn from INTERMEDIATE pool (numbers 4–6)
4. Continuous BKT updates during play
5. After question 20 (5 new questions):
   • p_know rises to 0.78
   • PC-BKT reclassifies Alex as ADVANCED
   • Question pool switches to numbers 7–10
```

---

## Project Structure

```
Inclusive-Mixed-Reality-Math-Experience-design-Using-ML-Based-Models/
│
├── With Model.py                  # Main application with BKT + PC-BKT adaptive engine
├── Without Model.py               # Computer vision demo (no ML adaptation)
├── pc_bkt_after_realdata.ipynb    # Jupyter notebook: model training & analysis
├── pc_bkt_model (1).pkl           # Pre-trained K-Means clustering model (k=3)
├── README.md                      # Project documentation
│
├── apple.png                      # Game asset: apple sprite
├── basket0.png … basket9.png      # Game assets: basket states (0–9 apples)
├── 1.MP3 … 10.MP3                 # Audio assets: counting sounds
├── Cheer.MP3                      # Audio asset: correct answer celebration
│
└── game_data/                     # Runtime-generated analytics (created on first run)
    ├── interactions.csv           # Per-question interaction log
    ├── sessions.csv               # Per-session summary statistics
    └── student_levels.json        # Persistent student profiles
```

---

## Future Enhancements

### Technical Improvements
- **Deep Knowledge Tracing (DKT)** — Replace BKT with LSTM-based model for richer temporal patterns
- **Speech recognition** — Integrate Vosk/Whisper for voice-command input (hands-free mode)
- **Sign language support** — Extend MediaPipe gesture vocabulary to ASL/BSL number recognition
- **Mobile deployment** — Touch-gesture fallback for tablet/smartphone platforms

### Platform Expansion
- **Unity/HoloLens port** — Full 3D mixed reality environment using Microsoft MRTK
- **Teacher dashboard** — Web-based analytics interface for classroom-wide insights
- **Automated retraining pipeline** — Continuous model improvement as interaction data accumulates
- **Multilingual audio** — Localized counting sounds and UI text

### Research Applications
- **A/B testing framework** — Compare BKT vs. DKT vs. rule-based difficulty systems
- **Longitudinal studies** — Track learning curves over weeks/months
- **Accessibility research** — Evaluate effectiveness for students with specific learning differences

---

## License

**Copyright © 2026 Anup806. All Rights Reserved.**

### Usage Restrictions

This software and associated documentation are provided for **portfolio demonstration and professional evaluation purposes only**.

**Prohibited without explicit written permission:**
- ❌ Copying, reproducing, or distributing this code
- ❌ Modifying or creating derivative works
- ❌ Using this software in commercial or personal projects
- ❌ Forking this repository for purposes other than evaluation

**Permitted actions:**
- ✅ Viewing the code and documentation for assessment purposes
- ✅ Evaluating this work for hiring, academic review, or research citation

### Intellectual Property Notice

This project contains proprietary implementations of:
- Custom BKT parameter tuning methodology
- PC-BKT feature engineering and K-Means integration
- Gesture-based interaction design patterns
- Multimodal feedback systems for inclusive learning

All algorithms, data structures, UI designs, and documentation are protected by copyright law.

### Contact for Permissions

For inquiries regarding licensing, collaboration, or academic citation:

- 📧 **Email:** raianup806@gmail.com
- 💼 **LinkedIn:** [Anup Rai](https://www.linkedin.com/in/anup-rai-095695343/)

---

**Project Status:** Active Development | Portfolio Demonstration  
**Repository:** [Anup806/Inclusive-Mixed-Reality-Math-Experience-design-Using-ML-Based-Models](https://github.com/Anup806/Inclusive-Mixed-Reality-Math-Experience-design-Using-ML-Based-Models)