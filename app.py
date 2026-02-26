from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import numpy as np
import pickle
from ultralytics import YOLO

from utilis import YOLO_Detection, drawPolygons, label_detection

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(r"/Users/np/Documents/Project_Parking_Lot_Robotics_Lab/yolov8n.pt")
model.to(device)

with open(r'/Users/np/Documents/Project_Parking_Lot_Robotics_Lab/Space_ROIs', 'rb') as f:
    posList = pickle.load(f)

global parking_status_data
parking_status_data = []

def generate_frames():
    global parking_status_data
    cap = cv2.VideoCapture(r"/Users/np/Documents/Project_Parking_Lot_Robotics_Lab/parking_space.mp4")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        boxes, classes, names = YOLO_Detection(model, frame)

        detection_points = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            detection_points.append((int(center_x), int(center_y)))

        frame, occupied_count, slot_statuses = drawPolygons(
            frame, 
            posList, 
            detection_points=detection_points, 
            polygon_color_inside=(0, 0, 255),   
            polygon_color_outside=(0, 255, 0)  
        )

        parking_status_data = slot_statuses

        available_count = len(posList) - occupied_count

        cv2.rectangle(frame, (int((width/2) - 200), 5), (int((width/2) - 40), 40), (250, 250, 250), -1)
        cv2.putText(frame, f"Fill Slots: {occupied_count}", (int((width/2) - 190), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (int(width/2), 5), (int((width/2) + 175), 40), (250, 250, 250), -1)
        cv2.putText(frame, f"Free Slots: {available_count}", (int((width/2) + 10), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, (50, 50, 50), 1, cv2.LINE_AA)

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            name = names[int(cls)]

            label_detection(frame=frame, text=str(name), left=x1, top=y1, bottom=x2, right=y2, tbox_color=(150, 150, 150))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    global parking_status_data
    return jsonify({"slots": parking_status_data})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)