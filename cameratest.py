import cv2

# Gunakan webcam pertama (id=0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka!")
else:
    print("Webcam berjaya dibuka!")

# Tangkap video dari webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal tangkap gambar dari webcam!")
        break

    # Tunjukkan gambar webcam
    cv2.imshow("Webcam", frame)

    # Tekan 'q' untuk berhenti
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan dan tutup
cap.release()
cv2.destroyAllWindows()
