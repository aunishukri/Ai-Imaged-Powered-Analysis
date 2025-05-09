import tensorflow as tf
import cv2
import numpy as np

# Load model dari folder
model = tf.keras.models.load_model("model.savedmodel")

# Label ikut urutan dalam Teachable Machine (anda boleh semak di metadata.json kalau ragu)
class_names = ["Made in Malaysia", "Made in Japan", "Made in China", "Made in UK"]

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]

    label = f"{class_names[class_id]} ({confidence*100:.1f}%)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Webcam Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
