from flask import Flask, render_template, Response, jsonify
import cv2
import time
import threading
from pathlib import Path
import torch
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "best.pt"  

CONF_THRES = 0.14

def pick_device():
    if torch.cuda.is_available():
        return 0

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
model = YOLO(str(WEIGHTS_PATH))

latest_jpeg = None
latest_lock = threading.Lock()

parking_status_data = []
parking_counts = {"empty": 0, "occupied": 0}

def _build_slots_from_detections(boxes, classes, names):
    dets = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cname = str(names[int(cls)]).strip().lower()

        if cname not in ("empty", "occupied"):
            continue

        dets.append({"cx": cx, "name": cname})

    if not dets:
        return [], 0, 0

    dets.sort(key=lambda d: d["cx"])

    slots = [d["name"] == "occupied" for d in dets]
    empty_count = sum(1 for d in dets if d["name"] == "empty")
    occ_count   = sum(1 for d in dets if d["name"] == "occupied")
    return slots, empty_count, occ_count

def camera_worker(cam_index=0):
    global latest_jpeg, parking_status_data, parking_counts

    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print("index permission error")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        results = model.predict(frame, conf=CONF_THRES, device=DEVICE, verbose=False)
        r = results[0]

        boxes = r.boxes.xyxy.tolist() if r.boxes is not None else []
        classes = r.boxes.cls.tolist() if r.boxes is not None else []
        names = r.names

        slots, empty_count, occ_count = _build_slots_from_detections(boxes, classes, names)
        parking_status_data = slots
        parking_counts = {"empty": empty_count, "occupied": occ_count}

        annotated = r.plot(labels=True, conf=True)

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        with latest_lock:
            latest_jpeg = buffer.tobytes()

        time.sleep(0.01)

threading.Thread(target=camera_worker, args=(0,), daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/snapshot")
def snapshot():
    with latest_lock:
        if latest_jpeg is None:
            return Response(status=204)
        data = latest_jpeg

    return Response(
        data,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.route("/status")
def status():
    return jsonify({"slots": parking_status_data, "counts": parking_counts})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)