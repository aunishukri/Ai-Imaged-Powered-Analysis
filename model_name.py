import cv2, pandas as pd, random, time
from pyzbar.pyzbar import decode
import csv

CSV_FILE     = 'barcode_metadata.csv'
CAMERA_INDEX = 0
MODEL_CLASSES= ['RFX-200X','MXR-310A','DRV-521M']

def read_current_barcode():
    try:
        b = open("current_barcode.txt").read().strip()
        return b if b else None
    except:
        return None

def classify_model_dummy(frame):
    return random.choice(MODEL_CLASSES)

def load_metadata(csv_path):
    with open(csv_path,newline='') as f:
        hdr = next(csv.reader(f))
    if len(hdr) == 1:
        cols = hdr[0].split(",")
        df = pd.read_csv(csv_path, names=cols, skiprows=1, dtype=str)
    else:
        df = pd.read_csv(csv_path, dtype=str)
    return df

def check_match(barcode_id, model_name):
    df = load_metadata(CSV_FILE)
    #print("Cols:", df.columns.tolist())  # uncomment utk debug
    return not df[
        (df['barcode_id']==barcode_id)&
        (df['model_name']==model_name)
    ].empty

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Gagal buka kamera"); return

    print("🔍 Tekan SPACE untuk scan & semak. ESC untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Checker", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            bc = read_current_barcode() or "<tiada barcode>"
            mn = classify_model_dummy(frame)
            print(f"\n📦 Barcode: {bc}\n📸 Model  : {mn}")
            if bc=="<tiada barcode>":
                print("⚠️ Tiada barcode!") ; status="NG"
            else:
                status = "OK" if check_match(bc,mn) else "NG"
                print(f"➡️ STATUS: {status}")
            time.sleep(2)

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
