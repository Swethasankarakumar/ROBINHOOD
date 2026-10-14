from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/SmartGossyculture_Boll_Detector/weights/best.pt")

# Predict on a test image
image_path = "cotton_dataset/val/images/test1.jpg"  # your image path
results = model.predict(source=image_path, show=True, save=True, conf=0.5)

for r in results:
    boxes = r.boxes
    for b in boxes:
        cls = int(b.cls)
        label = model.names[cls]
        conf = float(b.conf)
        print(f"🧠 Detected: {label}  (Confidence: {conf:.2f})")

print("✅ Prediction complete! Saved output in 'runs/detect/predict'.")
