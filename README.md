# Vision-Based Smart Conveyor Inventory Monitoring and Jam Prevention Agent

This project implements a **vision-based intelligent agent** for **real-time bottle tracking**, **inventory flow monitoring**, and **jam prevention** on a conveyor line. The system performs **bottle-only detection** using **YOLOv8 (COCO class 39)** and maintains consistent object identities using **ByteTrack tracking**. Based on the tracked bottle trajectories, the agent detects **PASS events**, identifies **REVERSE movement**, and triggers **JAM events** when bottles remain stationary in the conveyor band for a defined duration. A split-screen dashboard provides live KPIs, anomaly counts, jam risk prediction, and recommended actions, while key events are logged into a CSV file for reporting and analysis.

---

## Key Features
- **Perception (Vision Sensing):** YOLOv8 bottle detection + ByteTrack multi-object tracking  
- **Inventory Monitoring:** PASS counting when bottles move **Left → Right**  
- **Jam Event Logic:** JAM detected when a bottle stays in the middle band for **≥ 10 seconds** (configurable)  
- **Reverse Detection:** Identifies **Right → Left** return motion after PASS (zone-based + motion-based confirmation)  
- **Jam Risk Prediction (%):** Risk score computed from density-in-band, average speed, and recent jam frequency  
- **Action Output:** Decision output such as **STOP + ALARM**, **SLOW DOWN**, or **DIVERT REWORK**  
- **Split-Screen Dashboard:** Camera view + clean readable agent dashboard (KPIs + anomalies + actions)  
- **CSV Logging:** Writes to `inventory_agent_log.csv` when PASS count changes

---

## Intelligent Agent Rubric Mapping

### 1) Agent Perception (Input / Sensing)
- Input is a **live video stream** (webcam) or **video file**.
- The perception module detects bottles and outputs percepts in the form:
  - `(track_id, cx, cy, confidence)`

### 2) Decision Mechanism (Rules / Reasoning)
- **PASS Event:** counted when a tracked bottle transitions from **LEFT zone → RIGHT zone** (debounced for stability).
- **JAM Event:** triggered when a tracked bottle remains in the **middle band** continuously for at least `jam_seconds` (default 10s).
- **REVERSE Event:** triggered when a bottle returns from **RIGHT → LEFT** after a PASS, or when repeated negative motion is detected.
- **Jam Risk Predictor:** computes a risk percentage based on:
  - density of bottles in the band,
  - average motion speed,
  - recent jam occurrences (last 60 seconds).
- **Policy Actions (Decision Output):**
  - risk ≥ 80% → **STOP + ALARM**
  - risk ≥ 55% → **SLOW DOWN**
  - repeated reverse events → **DIVERT REWORK**

### 3) Action Execution (Output / Response)
- Executes actions by displaying alerts and recommendations on the dashboard.
- Logs system behaviour and count changes into CSV for traceability and reporting.

---

## Main Implementation File
All core logic (perception, tracking memory, decision rules, jam risk prediction, dashboard UI, and logging) is implemented in:

- **`bottle_tracking_gpu.py`**

Repository:  
https://github.com/sree-raam/Vision-Based-Smart-Conveyor-Inventory-Monitoring-and-Jam-Prevention-Agent

---

## Installation
Install required dependencies:

```bash
pip install ultralytics opencv-python numpy pandas
ytics opencv-python numpy pandas

How to Run
Run using webcam (default)python bottle_tracking_gpu.py
Run using a video filepython bottle_tracking_gpu.py --source path/to/video.mp4
Run using GPU (if supported)python bottle_tracking_gpu.py --device cuda:0
Exit the program by pressing q.

Important Parameters

--jam-seconds : jam threshold time in seconds (default 10.0)

--band : middle band width fraction for jam detection (default 0.06)

--debounce : stability debounce frames for zone switching (default 2)

--csv : output CSV log file (default inventory_agent_log.csv)

Example:python bottle_tracking_gpu.py --jam-seconds 8 --band 0.06 --debounce 2

Output

The system displays a split-screen monitoring window:

Left: annotated camera view with bottle IDs and band boundary lines

Right: dashboard showing PASS count, throughput, FPS, reverse/jam events, jam risk %, and last action

A CSV file is also generated for evaluation and reporting:

inventory_agent_log.csv
