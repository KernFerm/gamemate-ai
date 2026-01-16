# GameMate AI

> **Important Disclaimer**
>
> GameMate AI is **not an aimbot** and **not a cheat tool**.
>
> It does **not** automate gameplay, does **not** control mouse or keyboard input, does **not** inject into games, and does **not** read or write game memory.
>
> GameMate AI operates as a **user‑controlled desktop assistant** focused on system monitoring, visual overlays, and accessibility‑style enhancements that run alongside applications.

---

## Overview

GameMate AI is an intelligent desktop assistant designed to enhance the **user experience around gaming**, not gameplay itself.

It focuses on:

* System and performance visibility
* Profile‑based configuration
* Optional, non‑invasive visual overlays
* Stability, diagnostics, and reliability

All features are **opt‑in**, **user‑controlled**, and designed to respect platform and game policies.

---

## What GameMate AI DOES

* **Game Profile Management**
  YAML‑based profiles for organizing per‑game or per‑application settings.

* **Performance Monitoring**
  Real‑time tracking of FPS, CPU/GPU usage, memory usage, temperatures, and application health.

* **Privacy Controls**
  Local‑only data handling with user‑managed logging and retention settings.

* **Optional Visual Overlays**
  On‑screen overlays intended for clarity, visibility, and accessibility — fully controlled by the user.

---

## What GameMate AI DOES NOT DO

* ❌ No aim assist
* ❌ No recoil control
* ❌ No trigger automation
* ❌ No automated tracking input
* ❌ No mouse or keyboard control
* ❌ No memory reading or writing
* ❌ No game injection
* ❌ No anti‑cheat bypassing

GameMate AI **never interacts with game input or internal game data**.

---

## Modules

* **Vision / Scene**
  Visual analysis used exclusively for overlay rendering and UI assistance.

* **Gauge / Goggle / Scope**
  Optional visual enhancement modes (brightness, contrast, visibility assistance).

* **Tracker**
  Overlay‑only highlighting logic. No input control, no gameplay automation.

* **Performance Monitor**
  System and application telemetry.

* **Crash Reporter**
  Diagnostics and logging utilities.

* **Auto‑Updater / Version Manager**
  Keeps the application and optional models current.

---

## Installation

1. Install **Python 3.11+**
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download additional models:

   ```bash
   python scripts/download_models.py
   ```
4. Run the application:

   ```bash
   python main.py
   ```

---

## Configuration

Configuration is handled via `config.yaml`.

* Hardware acceleration
* Overlay preferences
* Performance limits
* Profile selection

Per‑application profiles are stored in `game_profiles/`.

---

## Supported Profiles

Profiles may reference popular games **only as configuration presets**.

They do **not** enable automation, cheating, or unfair advantages.

Users are responsible for ensuring compliance with the terms of service of any application or game they use alongside GameMate AI.

---

## Legal Notice

This project is an independent, experimental, and educational software project.

It is **not affiliated with, endorsed by, or produced by** any hardware manufacturer or game publisher.

All trademarks referenced are the property of their respective owners and are mentioned solely for descriptive purposes.
