import cv2
import torch

# Muatkan model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Guna model pre-trained YOLOv5

# Fungsi untuk deteksi objek
def detect_objects(frame):
    results = model(frame)  # Detect objek dalam frame
    results.show()  # Tunjuk hasil detection
    return results

# Setup webcam
cap = cv2.VideoCapture(0)  # 0 untuk webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi logo dan model
    results = detect_objects(frame)

    # Papar hasil dalam webcam
    cv2.imshow("Logo & Model Detection", results.imgs[0])  # Tunjuk frame yang telah dikesan

    if cv2.waitKey(1) & 0xFF == 27:  # Tekan 'Esc' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
