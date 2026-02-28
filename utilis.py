import cv2

def YOLO_Detection(model, frame, conf=0.15, device=None):
    results = model.predict(frame, conf=conf, device=device, verbose=False)

    r = results[0]
    boxes = r.boxes.xyxy.tolist() if r.boxes is not None else []
    classes = r.boxes.cls.tolist() if r.boxes is not None else []
    confs = r.boxes.conf.tolist() if r.boxes is not None else []
    names = r.names

    return results, boxes, classes, confs, names