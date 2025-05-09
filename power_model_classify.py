import cv2
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import pandas as pd

# === Load model dari Teachable Machine ===
model = tf.keras.models.load_model("keras_model.h5")

# === Senarai label ===
class_names = ['China', 'UK']  # Ubah ikut label anda di Teachable Machine

# === Fail log untuk simpan keputusan klasifikasi ===
log_file = "prediction_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["timestamp", "prediction"]).to_csv(log_file, index=False)

# === Fungsi pre-proses gambar dari webcam ===
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Teachable Machine biasanya 224x224
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# === Fungsi klasifikasi ===
def classify_frame(frame):
    processed = preprocess_frame(frame)
    prediction = model.predict(processed)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    return class_names[class_index], confidence

# === Fungsi simpan log ===
def log_prediction(label):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame([[now, label]], columns=["timestamp", "prediction"]).to_csv(log_file, mode="a", header=False, index=False)

# === Main webcam ===
cap = cv2.VideoCapture(0)
print("Tekan SPACE untuk klasifikasi, ESC untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Teachable Machine Detector", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    if key == 32:  # SPACEBAR
        label, confidence = classify_frame(frame)
        result_text = f"Made in {label} – OK ({confidence*100:.1f}%)"
        print(result_text)
        log_prediction(label)

        # Tunjuk hasil
        frame = cv2.putText(frame, result_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Teachable Machine Detector", frame)
        cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()
