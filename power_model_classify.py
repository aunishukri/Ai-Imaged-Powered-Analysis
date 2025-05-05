import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ==== Load your TensorFlow model ====
model = tf.keras.models.load_model('model/saved_model')  # Update path if needed

# ==== Class names must match your Roboflow dataset ====
class_names = ['Malaysia', 'UK']  # Adjust as needed

# ==== Output CSV file setup ====
log_file = 'classification_log.csv'
if not os.path.exists(log_file):
    pd.DataFrame(columns=['timestamp', 'prediction']).to_csv(log_file, index=False)

# ==== Function to preprocess frame ====
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Adjust if model uses different input size
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==== Function to classify frame ====
def classify_frame(frame):
    processed = preprocess_frame(frame)
    prediction = model.predict(processed)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    return class_names[class_id], confidence

# ==== Function to log results ====
def log_prediction(label):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame([[now, label]], columns=['timestamp', 'prediction'])
    df.to_csv(log_file, mode='a', header=False, index=False)

# ==== Main Webcam App ====
cap = cv2.VideoCapture(0)
print("Press SPACEBAR to classify, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show webcam feed
    cv2.imshow("Power Model Detector", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

    if key == 32:  # SPACEBAR
        label, confidence = classify_frame(frame)
        log_prediction(label)

        result_text = f"Made in {label} – OK"
        print(result_text)

        # Draw result on frame
        frame = cv2.putText(frame, result_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show result for 1 second
        cv2.imshow("Power Model Detector", frame)
        cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()
