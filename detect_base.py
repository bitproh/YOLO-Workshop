from ultralytics import YOLO

model = YOLO("yolo11n")

results = model("test_img_and_videos/dog.jpg")

result = results[0]

result.show() 

for box in result.boxes:
    cls_id = int(box.cls[0])        
    conf = float(box.conf[0])        
    label = model.names[cls_id]     
    print(f"Detected {label} with confidence {conf:.2f}")