import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)

print("Camera is active! Click the camera window and press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

   
    results = model(frame, stream=True)

    
    for r in results:
        annotated_frame = r.plot()

  
    cv2.imshow("NVIDIA RTX Vision Feed", annotated_frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
