import tensorflow as tf 
import cv2
import numpy as np
from keras.models import load_model

# Load model dan label
model = load_model("keras_Model.h5")
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Buka webcam
cap = cv2.VideoCapture(0)

print("Tekan SPACE untuk klasifikasi, ESC untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize dan normalisasi imej
    image = cv2.resize(frame, (224, 224))
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    key = cv2.waitKey(1)

    if key == 27:  # Tekan ESC untuk keluar
        break
    elif key == 32:  # Tekan SPACE untuk klasifikasi
        prediction = model.predict(image)
        index = np.argmax(prediction)
        label = class_names[index]
        confidence = prediction[0][index]
        print(f"Prediction: {label} ({confidence*100:.2f}%)")

        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", frame)
        cv2.waitKey(1000)

    cv2.imshow("Webcam", frame)

cap.release()
cv2.destroyAllWindows()
