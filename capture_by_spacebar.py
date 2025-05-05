import cv2
import os

# === Tetapan user ===
label_name = 'Malaysia'  # Tukar ke 'UK' untuk label lain
save_dir = f'dataset/{label_name}'  # Folder simpan gambar
max_images = 2000  # Berapa banyak gambar nak ambil

# === Buat folder kalau belum ada ===
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print(f"Tekan SPACEBAR untuk ambil gambar bagi label: {label_name}")
print("Tekan 'q' untuk keluar.")

counter = 0

while counter < max_images:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Ambil Gambar Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Berhenti manual oleh pengguna.")
        break

    if key == 32:  # SPACEBAR ditekan
        img_name = f"{label_name}_{counter+1:03}.jpg"
        img_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_path, frame)
        print(f"[{counter+1}/{max_images}] Disimpan: {img_path}")
        counter += 1

cap.release()
cv2.destroyAllWindows()
print("✅ Selesai ambil gambar.")
