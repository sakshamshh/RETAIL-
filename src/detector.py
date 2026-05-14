from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_people(frame):
    small = cv2.resize(frame, (480, 270))
    results = model(small, classes=[0], verbose=False, imgsz=480)
    boxes = []
    h_scale = frame.shape[0] / 270
    w_scale = frame.shape[1] / 480
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * w_scale)
            y1 = int(y1 * h_scale)
            x2 = int(x2 * w_scale)
            y2 = int(y2 * h_scale)
            conf = float(box.conf[0])
            if conf > 0.4:
                boxes.append((x1, y1, x2, y2, conf))
    return boxes

def draw_boxes(frame, boxes):
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame