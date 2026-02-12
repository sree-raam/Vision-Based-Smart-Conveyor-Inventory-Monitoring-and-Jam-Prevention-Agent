import argparse
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ============================================================
# ENVIRONMENT: THREADED VIDEO STREAM
# ============================================================
class VideoStream:
    """Threaded frame reader for smoother FPS on webcam/video."""
    def __init__(self, src: Union[int, str], width: int, height: int):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(src)

        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        if self.ret and self.frame is not None:
            return cv2.resize(self.frame, (self.width, self.height))
        return None

    def stop(self):
        self.running = False
        self.cap.release()


# ============================================================
# PERCEPTION: YOLOv8 + BYTETrack (BOTTLES ONLY)
# ============================================================
class BottlePerception:
    """
    Detects bottles using YOLOv8 (COCO class 39) and tracks using ByteTrack.
    Output percepts = [(track_id, cx, cy, confidence)]
    """
    BOTTLE_CLASS_ID = 39  # COCO bottle class

    def __init__(self, model_path: str, conf: float, tracker: str = "bytetrack.yaml", device: str = "cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker = tracker
        self.device = device

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, float]]]:
        r = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            device=self.device,
            conf=self.conf,
            classes=[self.BOTTLE_CLASS_ID],
            verbose=False
        )[0]

        annotated = frame.copy()
        percepts: List[Tuple[int, int, int, float]] = []

        if r.boxes is None or len(r.boxes) == 0 or r.boxes.id is None:
            return annotated, percepts

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), tid, cf in zip(boxes, ids, confs):
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            cx = int((x1i + x2i) / 2)
            cy = int((y1i + y2i) / 2)

            cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"ID:{tid}",
                (x1i, max(0, y1i - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

            percepts.append((int(tid), cx, cy, float(cf)))

        return annotated, percepts


# ============================================================
# TRACK MEMORY
# ============================================================
@dataclass
class TrackState:
    last_seen: float

    # PASS/REVERSE debounce (left/right only)
    stable_zone: str = "unknown"          # left/right
    stable_count: int = 0
    committed_zone: str = "unknown"       # left/right

    counted_pass: bool = False
    reverse_reported: bool = False
    jam_reported: bool = False

    last_cx: Optional[int] = None
    last_cy: Optional[int] = None
    last_t: Optional[float] = None
    speed_px_s: float = 0.0

    # JAM (band stuck)
    in_band_since: Optional[float] = None

    # reverse motion
    reverse_motion_count: int = 0

    # debug
    fsm_state: str = "NEW"


# ============================================================
# PREDICTION: JAM RISK SCORE
# ============================================================
@dataclass
class JamRiskPredictor:
    risk: float = 0.0
    jam_times: List[float] = None

    def __post_init__(self):
        if self.jam_times is None:
            self.jam_times = []

    def update(self, now: float, density: int, avg_speed: float, jam_happened: bool) -> float:
        if jam_happened:
            self.jam_times.append(now)

        self.jam_times = [t for t in self.jam_times if now - t <= 60.0]
        jam_rate = len(self.jam_times) / 60.0

        dens_score = min(1.0, density / 6.0)
        speed_score = 1.0 - min(1.0, avg_speed / 130.0)
        jam_score = min(1.0, jam_rate / 0.08)

        risk01 = 0.55 * dens_score + 0.30 * speed_score + 0.15 * jam_score

        self.risk = 0.85 * self.risk + 0.15 * (risk01 * 100.0)
        self.risk = float(np.clip(self.risk, 0.0, 100.0))
        return self.risk


# ============================================================
# AGENT: FIXED COUNTING + JAM + REVERSE
# ============================================================
class InventoryFlowAgent:
    """
    FIXED:
    - PASS counts on LEFT -> RIGHT (band ignored for debounce/commit)
    - JAM when stays in BAND >= jam_seconds
    - REVERSE zone/motion after PASS
    """

    def __init__(
        self,
        width: int,
        band_frac: float,
        debounce: int,
        jam_seconds: float,
        lost_timeout: float = 1.6,
        reverse_dx_thresh: int = 4,
        reverse_frames: int = 4,
    ):
        self.width = width
        self.band_frac = band_frac
        self.debounce = debounce
        self.jam_seconds = jam_seconds
        self.lost_timeout = lost_timeout

        self.reverse_dx_thresh = reverse_dx_thresh
        self.reverse_frames = reverse_frames

        self.pass_count = 0
        self.reverse_events = 0
        self.jam_events = 0

        self.mem: Dict[int, TrackState] = {}
        self.events: List[str] = []
        self._pass_times: List[float] = []

        self.predictor = JamRiskPredictor()
        self.last_action: str = "NONE"
        self.last_action_t: float = 0.0

        self._recompute_lines()

    def _recompute_lines(self):
        mid = self.width // 2
        band_px = int(self.width * self.band_frac)
        self.enter_x = mid - band_px
        self.exit_x = mid + band_px

    def _observe_side(self, cx: int) -> str:
        if cx < self.enter_x:
            return "left"
        if cx > self.exit_x:
            return "right"
        return "band"

    def throughput_per_min(self, now: float, window_sec: float = 30.0) -> float:
        self._pass_times = [t for t in self._pass_times if now - t <= window_sec]
        return (len(self._pass_times) / window_sec) * 60.0

    def compute_features(self) -> Tuple[int, float]:
        density = 0
        speeds = []
        for st in self.mem.values():
            if st.last_cx is None:
                continue
            if self._observe_side(st.last_cx) == "band":
                density += 1
            if st.speed_px_s > 0:
                speeds.append(st.speed_px_s)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        return density, avg_speed

    def decide_actions(self, now: float, risk: float):
        if now - self.last_action_t < 1.0:
            return
        action = "NONE"
        if risk >= 80.0:
            action = "STOP + ALARM"
        elif risk >= 55.0:
            action = "SLOW DOWN"
        elif self.reverse_events > 0 and (self.reverse_events % 2 == 0):
            action = "DIVERT REWORK"

        if action != "NONE":
            self.last_action = action
            self.last_action_t = now
            self._log_event(f"ACTION | {action}")

    def update(self, tid: int, cx: int, cy: int, now: float):
        if tid not in self.mem:
            self.mem[tid] = TrackState(last_seen=now)
        st = self.mem[tid]
        st.last_seen = now

        # speed + dx
        dx = 0
        if st.last_cx is not None and st.last_t is not None:
            dt = max(1e-6, now - st.last_t)
            dx = cx - st.last_cx
            st.speed_px_s = 0.7 * st.speed_px_s + 0.3 * (abs(dx) / dt)

        st.last_cx, st.last_cy, st.last_t = cx, cy, now
        side = self._observe_side(cx)

        # ---------------------------
        # JAM (band stuck) - DO NOT break counting logic
        # ---------------------------
        if side == "band":
            if st.in_band_since is None:
                st.in_band_since = now
            else:
                if (not st.jam_reported) and ((now - st.in_band_since) >= self.jam_seconds):
                    st.jam_reported = True
                    self.jam_events += 1
                    self._log_event(f"JAM | ID:{tid}")
            # Do NOT update debounce/commit while in band
            st.fsm_state = "IN_BAND" if not st.jam_reported else "JAM_DETECTED"
            return
        else:
            st.in_band_since = None

        # ---------------------------
        # Debounce stable zone (left/right only)
        # ---------------------------
        if side == st.stable_zone:
            st.stable_count += 1
        else:
            st.stable_zone = side
            st.stable_count = 1

        if st.stable_count < self.debounce:
            st.fsm_state = f"DEBOUNCE_{side.upper()}"
            return

        # Commit initial zone
        if st.committed_zone == "unknown":
            st.committed_zone = st.stable_zone
            st.fsm_state = f"LOCKED_{st.committed_zone.upper()}"
            return

        # Zone change -> PASS / REVERSE(zone)
        if st.stable_zone != st.committed_zone:
            # PASS: LEFT -> RIGHT
            if st.committed_zone == "left" and st.stable_zone == "right" and (not st.counted_pass):
                st.counted_pass = True
                self.pass_count += 1
                self._pass_times.append(now)
                st.fsm_state = "PASS_CONFIRMED"
                self._log_event(f"PASS | ID:{tid}")

            # REVERSE(zone): RIGHT -> LEFT after PASS
            elif st.committed_zone == "right" and st.stable_zone == "left":
                if st.counted_pass and (not st.reverse_reported):
                    st.reverse_reported = True
                    self.reverse_events += 1
                    st.fsm_state = "REVERSE_DETECTED_ZONE"
                    self._log_event(f"REVERSE | ID:{tid}")

            st.committed_zone = st.stable_zone

        # REVERSE (motion-based) after PASS
        if st.counted_pass and (not st.reverse_reported):
            if dx <= -self.reverse_dx_thresh:
                st.reverse_motion_count += 1
            else:
                st.reverse_motion_count = max(0, st.reverse_motion_count - 1)

            if st.reverse_motion_count >= self.reverse_frames:
                st.reverse_reported = True
                self.reverse_events += 1
                st.fsm_state = "REVERSE_DETECTED_MOTION"
                self._log_event(f"REVERSE | ID:{tid}")
        else:
            if st.counted_pass:
                st.fsm_state = "TRACKING_AFTER_PASS"

    def cleanup(self, now: float):
        remove_ids = [tid for tid, st in self.mem.items() if now - st.last_seen > self.lost_timeout]
        for tid in remove_ids:
            del self.mem[tid]

    def _log_event(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"{ts} {msg}")
        self.events = self.events[-8:]


# ============================================================
# CSV LOGGER
# ============================================================
class EventLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=[
                "Timestamp",
                "PASS_Count",
                "Reverse_Events",
                "JAM_Events",
                "Throughput_items_min",
                "Density_in_band",
                "Avg_speed_px_s",
                "Jam_risk_pct",
                "Last_action",
                "Note"
            ]).to_csv(csv_path, index=False)
        self.last_count: Optional[int] = None

    def log_if_count_changed(self, agent: InventoryFlowAgent, tput: float, density: int, avg_speed: float, risk: float):
        if self.last_count != agent.pass_count:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pd.DataFrame([[
                ts,
                agent.pass_count,
                agent.reverse_events,
                agent.jam_events,
                float(tput),
                int(density),
                float(avg_speed),
                float(risk),
                agent.last_action,
                "COUNT_CHANGED"
            ]], columns=[
                "Timestamp", "PASS_Count", "Reverse_Events", "JAM_Events",
                "Throughput_items_min", "Density_in_band", "Avg_speed_px_s",
                "Jam_risk_pct", "Last_action", "Note"
            ]).to_csv(self.csv_path, mode="a", header=False, index=False)
            self.last_count = agent.pass_count


# ============================================================
# DASHBOARD UI (your working layout)
# ============================================================
def make_dashboard_big(agent, fps, tput, density, avg_speed, risk, width, height):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (16, 16, 16)

    WHITE = (245, 245, 245)
    MUTED = (185, 185, 185)
    BORDER = (95, 95, 95)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    PAD = 16
    GAP = 10
    INNER = 18

    header_h = 82
    bottom_pad = 12

    x1, x2 = PAD, width - PAD
    usable_h = height - header_h - bottom_pad - 2 * GAP

    # Give more space to the bottom box (Prediction & Actions)
    box1_h = int(usable_h * 0.28)
    box2_h = int(usable_h * 0.28)
    box3_h = usable_h - box1_h - box2_h

    top1, bot1 = header_h, header_h + box1_h
    top2, bot2 = bot1 + GAP, bot1 + GAP + box2_h
    top3, bot3 = bot2 + GAP, bot2 + GAP + box3_h

    TITLE_S = 0.98
    SUB_S = 0.62
    SEC_S = 0.80
    LABEL_S = 0.68
    VALUE_S = 0.76
    BIG_S = 0.86
    SMALL_S = 0.56

    VALUE_COL = int(width * 0.62)
    RIGHT_SAFE = width - PAD - 10

    def put(x, y, text, scale=0.85, color=WHITE, thick=2):
        cv2.putText(panel, text, (x + 2, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(panel, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def right_text(y, text, scale=0.95, color=WHITE, thick=2):
        (tw, _), _b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = max(VALUE_COL, RIGHT_SAFE - tw)
        put(x, y, text, scale=scale, color=color, thick=thick)

    def box(xa, ya, xb, yb, title):
        cv2.rectangle(panel, (xa, ya), (xb, yb), BORDER, 2)
        cv2.rectangle(panel, (xa + 2, ya + 2), (xb - 2, ya + 38), (28, 28, 28), -1)
        put(xa + 12, ya + 26, title, scale=SEC_S, color=CYAN, thick=2)

    # Header
    put(PAD, 40, "AGENT DASHBOARD", scale=TITLE_S, color=WHITE, thick=2)
    put(PAD, 66, "Big UI Mode (Clean & Readable)", scale=SUB_S, color=MUTED, thick=1)

    # KPIs
    box(x1, top1, x2, bot1, "LIVE KPIs")
    y = top1 + 60
    step = max(22, int((box1_h - 70) / 3))
    put(x1 + INNER, y, "PASS COUNT", scale=LABEL_S, color=MUTED, thick=2)
    right_text(y, str(agent.pass_count), scale=BIG_S, color=RED, thick=2)

    y += step
    put(x1 + INNER, y, "THROUGHPUT", scale=LABEL_S, color=MUTED, thick=2)
    right_text(y, f"{tput:.1f} items/min", scale=VALUE_S, color=WHITE, thick=2)

    y += step
    put(x1 + INNER, y, "FPS", scale=LABEL_S, color=MUTED, thick=2)
    fps_color = GREEN if fps >= 15 else (YELLOW if fps >= 8 else RED)
    right_text(y, f"{fps:.2f}", scale=VALUE_S, color=fps_color, thick=2)

    # Anomalies
    box(x1, top2, x2, bot2, "ANOMALIES")
    y = top2 + 60
    step2 = max(24, int((box2_h - 70) / 2))
    put(x1 + INNER, y, "REVERSE EVENTS", scale=LABEL_S, color=MUTED, thick=2)
    right_text(y, str(agent.reverse_events), scale=VALUE_S, color=YELLOW, thick=2)

    y += step2
    put(x1 + INNER, y, "JAM EVENTS", scale=LABEL_S, color=MUTED, thick=2)
    right_text(y, str(agent.jam_events), scale=VALUE_S, color=YELLOW, thick=2)

    # Prediction & Actions
    box(x1, top3, x2, bot3, "PREDICTION & ACTIONS")

    # Smaller "Recent" area so nothing overlaps
    events_h = 38
    metrics_bottom = bot3 - events_h

    start_y = top3 + 56
    available = max(80, metrics_bottom - start_y - 6)
    mstep = max(20, int(available / 4))

    y = start_y
    put(x1 + INNER, y, "DENSITY (BAND)", scale=LABEL_S, color=MUTED, thick=2)
    right_text(y, str(density), scale=VALUE_S, color=WHITE, thick=2)

    y += mstep
    put(x1 + INNER, y, "AVG SPEED", scale=LABEL_S, color=MUTED, thick=2)
    right_text(y, f"{avg_speed:.1f} px/s", scale=VALUE_S, color=WHITE, thick=2)

    y += mstep
    put(x1 + INNER, y, "JAM RISK", scale=LABEL_S, color=MUTED, thick=2)
    risk_color = GREEN if risk < 35 else (YELLOW if risk < 70 else RED)
    right_text(y, f"{risk:.0f}%", scale=VALUE_S, color=risk_color, thick=2)

    y += mstep
    put(x1 + INNER, y, "LAST ACTION", scale=LABEL_S, color=MUTED, thick=2)
    action_color = CYAN if agent.last_action != "NONE" else MUTED
    right_text(y, agent.last_action, scale=0.70, color=action_color, thick=2)

    cv2.line(panel, (x1 + 8, metrics_bottom), (x2 - 8, metrics_bottom), (60, 60, 60), 1)

    ey = metrics_bottom + 22
    put(x1 + INNER, ey, "Recent:", scale=SMALL_S, color=MUTED, thick=1)
    last = agent.events[-1] if agent.events else "None"
    if len(last) > 56:
        last = last[:53] + "..."
    put(x1 + INNER + 78, ey, last, scale=SMALL_S, color=WHITE, thick=1)

    return panel



# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Inventory Agent (Split Screen Camera + Perfect Dashboard)")
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)

    parser.add_argument("--band", type=float, default=0.06)  # more reliable than 0.02
    parser.add_argument("--debounce", type=int, default=2)
    parser.add_argument("--jam-seconds", type=float, default=10.0)

    parser.add_argument("--csv", default="inventory_agent_log.csv")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])

    parser.add_argument("--dash-width", type=int, default=820)
    parser.add_argument("--scale", type=float, default=1.0)

    parser.add_argument("--reverse-dx", type=int, default=4)
    parser.add_argument("--reverse-frames", type=int, default=4)

    args = parser.parse_args()
    src = int(args.source) if str(args.source).isdigit() else args.source

    stream = VideoStream(src, args.width, args.height)
    perception = BottlePerception(args.model, conf=args.conf, device=args.device)

    agent = InventoryFlowAgent(
        width=args.width,
        band_frac=args.band,
        debounce=args.debounce,
        jam_seconds=args.jam_seconds,
        reverse_dx_thresh=args.reverse_dx,
        reverse_frames=args.reverse_frames
    )
    logger = EventLogger(args.csv)

    prev_t = time.time()

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            now = time.time()
            annotated, percepts = perception.infer(frame)

            h, w = annotated.shape[:2]
            cv2.line(annotated, (agent.enter_x, 0), (agent.enter_x, h), (255, 255, 0), 2)
            cv2.line(annotated, (agent.exit_x, 0), (agent.exit_x, h), (255, 255, 0), 2)

            for tid, cx, cy, cf in percepts:
                agent.update(tid, cx, cy, now)
                st = agent.mem.get(tid)
                if st:
                    cv2.putText(
                        annotated, st.fsm_state,
                        (cx + 6, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2, cv2.LINE_AA
                    )

            agent.cleanup(now)

            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now

            density, avg_speed = agent.compute_features()
            jam_happened = bool(agent.events) and ("JAM |" in agent.events[-1])
            risk = agent.predictor.update(now, density=density, avg_speed=avg_speed, jam_happened=jam_happened)
            agent.decide_actions(now, risk)
            tput = agent.throughput_per_min(now=now, window_sec=30.0)

            logger.log_if_count_changed(agent, tput, density, avg_speed, risk)

            dashboard = make_dashboard_big(agent, fps, tput, density, avg_speed, risk,
                                           width=args.dash_width, height=h)

            split_view = np.hstack([annotated, dashboard])
            cv2.line(split_view, (w, 0), (w, split_view.shape[0]), (120, 120, 120), 3)

            if args.scale != 1.0:
                split_view = cv2.resize(
                    split_view,
                    (int(split_view.shape[1] * args.scale), int(split_view.shape[0] * args.scale))
                )

            cv2.imshow("Inventory Agent (Split View)", split_view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        logging.info(f"CSV saved: {args.csv}")

    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
