import cv2
from ultralytics import YOLO

# Load YOLOv11n model
model = YOLO("yolo11n")

cap = cv2.VideoCapture(0)

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("YOLOv11n Video Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()