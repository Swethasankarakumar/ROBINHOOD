import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/SmartGossyculture_Boll_Detector/weights/best.pt")

cap = cv2.VideoCapture(0)  # 0 for default webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, stream=True)
    for r in results:
        annotated = r.plot()
        cv2.imshow("Smart Gossyculture - Live Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
