import cv2
from ultralytics import YOLO

# 1. Load the lightweight YOLO model
model = YOLO("yolov8n.pt")

# 2. Connect to your laptop webcam (0 is built-in camera)
cap = cv2.VideoCapture(0)

print("Camera is active! Click the camera window and press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 3. Detect objects in the frame
    results = model(frame, stream=True)

    # 4. Draw boxes around detected items
    for r in results:
        annotated_frame = r.plot()

    # 5. Show the window
    cv2.imshow("NVIDIA RTX Vision Feed", annotated_frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()