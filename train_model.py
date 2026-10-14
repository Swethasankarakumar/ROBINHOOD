from ultralytics import YOLO
import torch

print("✅ GPU Available:", torch.cuda.is_available())

# Load pre-trained YOLOv8 small model
model = YOLO("yolov8s.pt")

# Train your dataset
model.train(
    data="cotton_dataset/data.yaml",   # path to YAML
    epochs=10,                         # number of training epochs
    imgsz=640,                         # image size
    batch=8,                           # adjust based on RAM/GPU
    name="SmartGossyculture_Boll_Detector"
)

print("🎯 Training Complete! Check the 'runs/detect/SmartGossyculture_Boll_Detector' folder for results.")
