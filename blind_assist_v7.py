"""
╔══════════════════════════════════════════════════════════════╗
║          BLIND ASSIST — V7                                   ║
║  Extra Feature: all detectioon save in csv file              ║
║  CSV file: detection_log.csv                                 ║
║  Hotkeys:  Q=Quit | R=Read Text                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import time
import threading
import queue
import subprocess
import os
import sys
import csv
from datetime import datetime
from ultralytics import YOLO

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


CONF_THRESHOLD  = 0.6
DANGER_COOLDOWN = 2.5
NORMAL_COOLDOWN = 4.0
CLEAR_PATH_CD   = 7.0
CAMERA_INDEX    = 0
LOG_FILE        = "detection_log.csv"

FOCAL_LENGTH = 615


ALLOWED_CLASSES = {
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "dog", "cat", "bottle", "cell phone", "chair", "traffic light",
    "stop sign", "backpack", "suitcase", "umbrella", "book", "laptop"
}

REAL_WIDTHS = {
    "car": 1.8, "bus": 2.5, "truck": 2.4, "motorcycle": 0.8,
    "train": 3.0, "bicycle": 0.6,
    "person": 0.5, "dog": 0.4, "cat": 0.3, "horse": 1.2, "cow": 1.0,
    "chair": 0.5, "couch": 1.8, "dining table": 1.2, "bed": 1.4,
    "toilet": 0.4, "sink": 0.5, "refrigerator": 0.7, "bench": 1.2,
    "traffic light": 0.3, "stop sign": 0.6, "fire hydrant": 0.3, "pole": 0.15,
    "bottle": 0.08, "cup": 0.09, "bowl": 0.2, "laptop": 0.35,
    "tv": 1.0, "backpack": 0.35, "handbag": 0.3, "suitcase": 0.5,
    "umbrella": 1.0, "cell phone": 0.07, "book": 0.2,
}
DEFAULT_REAL_WIDTH = 0.4

HIGH_DANGER = {
    "car", "bus", "truck", "motorcycle", "train", "bicycle", "pole",
}


#  CSV LOGGER


csv_file   = open(LOG_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "object", "confidence", "distance_m", "direction", "danger_tier"])

def log_detection(label, conf, dist_m, direction, tier):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([ts, label, round(conf, 2), round(dist_m, 2), direction, tier])
    csv_file.flush()


#  TTS

_tts_queue = queue.Queue()

def _tts_worker():
    while True:
        text = _tts_queue.get()
        if text is None:
            break
        try:
            safe = text.replace("'", " ")
            ps   = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = 1; $s.Speak('{safe}');"
            )
            subprocess.run(["powershell", "-Command", ps],
                           capture_output=True, timeout=15)
        except Exception as e:
            print(f"[TTS Error] {e}")
        _tts_queue.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text: str):
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
            _tts_queue.task_done()
        except queue.Empty:
            break
    _tts_queue.put(text)
    print(f"[TTS] >> {text}")

#  DISTANCE + DIRECTION HELPERS

def estimate_distance(label, bbox_w):
    real_w = REAL_WIDTHS.get(label, DEFAULT_REAL_WIDTH)
    if bbox_w < 1:
        return 99.0
    return round((real_w * FOCAL_LENGTH) / bbox_w, 2)

def tts_distance(metres):
    if metres < 1.0:
        return f"{int(round(metres * 100))} centimetres"
    elif metres == int(metres):
        m = int(metres)
        return f"{m} metre" if m == 1 else f"{m} metres"
    else:
        w = int(metres)
        d = int(round((metres - w) * 10))
        return f"{w} point {d} metres"

def display_distance(metres):
    if metres < 1.0:
        return f"{int(metres*100)} cm"
    return f"{metres:.1f} m"

def get_tier(label):
    return "HIGH" if label in HIGH_DANGER else "NORMAL"

def get_direction(cx, fw):
    if cx < fw / 3:      return "left"
    if cx > 2 * fw / 3:  return "right"
    return "center"

def make_message(label, dist_m, direction, tier):
    d = tts_distance(dist_m)
    if tier == "HIGH":
        if dist_m < 1.0:
            return f"Warning! {label} on your {direction}, only {d} away! Stop!"
        elif dist_m < 3.0:
            return f"Caution! {label} on your {direction}, only {d} away!"
        else:
            return f"Caution! {label} on your {direction}, {d} away"
    else:
        if direction == "center":
            return f"{label} ahead, {d} away"
        return f"{label} on your {direction}, {d} away"


#  OCR


def read_text(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, config=custom_config).strip()
    speak(f"Text detected: {text}" if text else "No readable text found")


#  INIT — DONO MODELS LOAD

print("=" * 60)
print("  BLIND ASSIST V7 — TRANSFER LEARNING EDITION")
print(f"  Detection log: {LOG_FILE}")
print("=" * 60)

# Transfer learned custom model (road obstacles)
model_custom = YOLO(r"C:\Users\mauli\Downloads\runs\detect\blind_assist_finetuned\weights\best.pt")

# Original COCO model (phone, bottle, etc)
model_coco = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Camera nahi mila!")
    sys.exit(1)

speak("Blind assist started. Transfer learning model active.")

last_spoken     = {}
last_speak_time = 0.0
ocr_busy        = False
total_logged    = 0

print(f"\nits already on ! Data is logging: {LOG_FILE}\n")


#  MAIN LOOP


while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # both models result
    results_custom = model_custom(frame, verbose=False)
    results_coco   = model_coco(frame, verbose=False)

    frame_w   = frame.shape[1]
    current_t = time.time()

    detections = []

    # Custom model detections
    for box in results_custom[0].boxes:
        cls   = int(box.cls[0])
        label = results_custom[0].names[cls]
        conf  = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.xyxy[0]
        width    = float(x2 - x1)
        center_x = float((x1 + x2) / 2)
        dist_m   = estimate_distance(label, width)
        tier     = get_tier(label)
        direction = get_direction(center_x, frame_w)
        log_detection(label, conf, dist_m, direction, tier)
        total_logged += 1
        detections.append({
            "label": label, "tier": tier,
            "dist_m": dist_m, "direction": direction,
            "conf": conf,
        })

    # COCO model detections — only ALLOWED_CLASSES
    for box in results_coco[0].boxes:
        cls   = int(box.cls[0])
        label = results_coco[0].names[cls]
        conf  = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        if label not in ALLOWED_CLASSES:  
            continue
        x1, y1, x2, y2 = box.xyxy[0]
        width    = float(x2 - x1)
        center_x = float((x1 + x2) / 2)
        dist_m   = estimate_distance(label, width)
        tier     = get_tier(label)
        direction = get_direction(center_x, frame_w)
        log_detection(label, conf, dist_m, direction, tier)
        total_logged += 1
        detections.append({
            "label": label, "tier": tier,
            "dist_m": dist_m, "direction": direction,
            "conf": conf,
        })


    seen = set()
    unique_detections = []
    for d in detections:
        key = f"{d['label']}_{d['direction']}"
        if key not in seen:
            seen.add(key)
            unique_detections.append(d)
    detections = unique_detections

    detections.sort(key=lambda d: (0 if d["tier"] == "HIGH" else 1, d["dist_m"]))

    # Speak
    spoken_this_frame = False
    for det in detections:
        if det["dist_m"] > 15.0 and det["tier"] != "HIGH":
            continue
        cooldown = DANGER_COOLDOWN if det["tier"] == "HIGH" else NORMAL_COOLDOWN
        key = f'{det["tier"]}_{det["label"]}_{det["direction"]}'
        if current_t - last_spoken.get(key, 0) > cooldown:
            speak(make_message(det["label"], det["dist_m"], det["direction"], det["tier"]))
            last_spoken[key] = current_t
            last_speak_time  = current_t
            spoken_this_frame = True
            break

    if not spoken_this_frame and not detections:
        if current_t - last_speak_time > CLEAR_PATH_CD:
            speak("Path is clear")
            last_speak_time = current_t

    # Display
    fps       = 1.0 / max(time.time() - loop_start, 1e-6)
    annotated = results_coco[0].plot()
    top       = detections[0] if detections else None

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (480, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

    cv2.putText(annotated, f"FPS: {int(fps)}  |  Logged: {total_logged} detections",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if top:
        tier_color = (0, 0, 255) if top["tier"] == "HIGH" else (0, 200, 0)
        cv2.putText(annotated,
                    f"{top['label']} | {display_distance(top['dist_m'])} | {top['direction']} [{top['tier']}]",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tier_color, 2)
    cv2.putText(annotated, "R=Text  Q=Quit  | Log: detection_log.csv",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.imshow("Blind Assist V7 — Transfer Learning Edition", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        speak("Shutting down. Data saved.")
        time.sleep(1.5)
        break
    elif key == ord('r') and not ocr_busy:
        ocr_busy = True
        def _ot(f):
            global ocr_busy
            read_text(f)
            ocr_busy = False
        threading.Thread(target=_ot, args=(frame.copy(),), daemon=True).start()

# Cleanup
csv_file.close()
_tts_queue.put(None)
cap.release()
cv2.destroyAllWindows()
print(f"\nTotal {total_logged} detections logged → {LOG_FILE}")
print("now we will analytics one to show all the graphs")
