import cv2
import numpy as np

def YOLO_Detection(model, frame, conf=0.35):
    results = model.predict(frame, conf=conf, classes=[0,2])
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    return boxes, classes, names

def label_detection(frame, text, left, top, bottom, right, tbox_color=(30, 155, 50), fontFace=1, fontScale=0.8, fontThickness=1):
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 2)
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    y_adjust = 10
    cv2.rectangle(frame, (int(left), int(top) - text_h - y_adjust), (int(left) + text_w + y_adjust, int(top)), tbox_color, -1)
    cv2.putText(frame, text, (int(left) + 5, int(top) - 5), fontFace, fontScale, (255, 255, 255), fontThickness, cv2.LINE_AA)

def drawPolygons(frame, points_list, detection_points=None, polygon_color_inside=(30, 205, 50), polygon_color_outside=(30, 50, 180), alpha=0.5):
    overlay = frame.copy()
    occupied_polygons = 0

    slot_statuses = [] 

    for area in points_list:
        area_np = np.array(area, np.int32)
        
        if detection_points:

            is_inside = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points)
        else:
            is_inside = False
            
        color = polygon_color_inside if is_inside else polygon_color_outside

        slot_statuses.append(is_inside)

        if is_inside:
            occupied_polygons += 1

        cv2.fillPoly(overlay, [area_np], color)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame, occupied_polygons, slot_statuses