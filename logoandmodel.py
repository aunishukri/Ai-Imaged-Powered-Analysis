import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import os
from roboflow import Roboflow

# ==== Set up Roboflow API and model ====
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE_NAME").project("YOUR_PROJECT_NAME")
model = project.version("1").model  # Replace with your model version

# ==== Set up class names for logo classification ====
logo_class_names = ['OK', 'NG']  # Update this based on your Roboflow classes

# ==== Output CSV file setup for logging predictions ====
log_file = 'classification_log.csv'
if not os.path.exists(log_file):
    pd.DataFrame(columns=['timestamp', 'prediction']).to_csv(log_file, index=False)

# ==== Function to preprocess frame (resize and normalize image) ====
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Adjust if your model expects a different size
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ==== Function to classify logo and model name from the frame ====
def classify_frame(frame):
    processed = preprocess_frame(frame)
    prediction = model.predict(processed)
    
    # Assuming Roboflow's model returns prediction with labels (for logo)
    class_id = prediction["class"]
    confidence = prediction["confidence"]
    
    return logo_class_names[class_id], confidence

# ==== Function to log predictions to CSV ====
def log_prediction(label):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame([[now, label]], columns=['timestamp', 'prediction'])
    df.to_csv(log_file, mode='a', header=False, index=False)

# ==== Main loop for webcam ====
cap = cv2.VideoCapture(0)
print("Press SPACEBAR to classify, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display webcam feed
    cv2.imshow("Logo & Model Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key to exit
        break

    if key == 32:  # SPACEBAR key to classify the image
        label, confidence = classify_frame(frame)
        log_prediction(label)
        print(f"Detected: {label} ({confidence*100:.2f}%)")

        # Draw result on the frame
        result_text = f"{label} ({confidence*100:.1f}%)"
        frame = cv2.putText(frame, result_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Logo & Model Detection", frame)
        cv2.waitKey(1000)  # Pause for 1 second before next frame

cap.release()
cv2.destroyAllWindows()
print("Webcam closed. Exiting...")