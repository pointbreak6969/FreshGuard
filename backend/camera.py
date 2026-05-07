from ultralytics import YOLO
import cv2

model = YOLO('yolo11x.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    results = model(frame, imgsz=640, conf=0.4)
    PRODUCE = {
    "apple", "banana", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "cake",
    "sandwich", "donut", "tomato", "potato",
}
    for box in results[0].boxes:
        class_id = int(box.cls)
        label = model.names[class_id]
        confidence = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
        frame,
        f"{label} {confidence:.0%}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
        )
        print(f"Detected {label} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
    annotated_frame = results[0].plot()
    

    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()