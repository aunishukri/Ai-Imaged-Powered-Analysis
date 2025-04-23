import streamlit as st
import cv2
import torch
import time

st.title("🚀 Logo & Model Detection")

# Muatkan model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Fungsi untuk deteksi objek
def detect_objects(frame):
    results = model(frame)  # Detect objek dalam frame
    return results

# Setup webcam
cap = cv2.VideoCapture(0)

# Auto refresh setiap 2 saat
st.markdown("⏳ Auto-refresh setiap 2 saat...")
placeholder = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi logo dan model
    results = detect_objects(frame)

    # Render hasil detection pada Streamlit
    with placeholder.container():
        st.image(results.imgs[0], channels="BGR")

    time.sleep(2)  # refresh setiap 2 saat

cap.release()
cv2.destroyAllWindows()
