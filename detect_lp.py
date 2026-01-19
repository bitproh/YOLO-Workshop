import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os
import re

# ---------------- CONFIG ----------------
MODEL_PATH = "license_plate_detector.pt"     # your trained YOLO LP model
SOURCE = "traffic.mp4"               # 0 = webcam, or video path
CONF = 0.3
CSV_FILE = "plates.csv"
DEVICE = "cpu"            # change to 0 for GPU
# ---------------------------------------

# Load model
model = YOLO(MODEL_PATH)

# OCR
reader = easyocr.Reader(['en'], gpu=(DEVICE != "cpu"))

# Load existing CSV to avoid duplicates
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    logged_plates = set(df["license_number"].astype(str))
else:
    df = pd.DataFrame(columns=["license_number", "timestamp"])
    logged_plates = set()

cap = cv2.VideoCapture(SOURCE)

print("[INFO] License plate detection started...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF, device=DEVICE, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # OCR
            text_list = reader.readtext(plate_crop, detail=0)
            if not text_list:
                continue

            plate = text_list[0].upper().replace(" ", "")

            # Basic filter to remove garbage OCR
            if not re.match(r'^[A-Z0-9]{6,12}$', plate):
                continue

            # Log only once
            if plate not in logged_plates:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"[DETECTED] {plate} at {timestamp}")

                logged_plates.add(plate)
                df.loc[len(df)] = [plate, timestamp]
                df.to_csv(CSV_FILE, index=False)

cap.release()
print("\n[INFO] Detection stopped. CSV updated.")
