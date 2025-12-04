import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

"""
Step 1: Deteksi wajah + ekstraksi sinyal rPPG mentah dari webcam.

Fungsi:
- Menangkap video dari webcam.
- Mendeteksi wajah dengan MediaPipe.
- Menentukan ROI di dahi.
- Mengambil rata-rata intensitas kanal hijau (Green) di ROI setiap frame.
- Menyimpan sinyal mentah ke buffer untuk diproses di tahap berikutnya.

Tekan tombol 'q' di jendela video untuk keluar.
"""

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection


def get_forehead_roi(frame: np.ndarray, detection: mp_face_detection.FaceDetection) -> tuple:
    """Hitung koordinat ROI dahi berdasarkan bounding box wajah.

    ROI diambil di bagian atas wajah agar lebih stabil dan minim gerakan mulut.
    Mengembalikan tuple (x1, y1, x2, y2).
    """
    h, w, _ = frame.shape
    bboxC = detection.location_data.relative_bounding_box

    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    w_box = int(bboxC.width * w)
    h_box = int(bboxC.height * h)

    # ROI dahi: bagian atas bounding box wajah
    x1 = max(x + int(0.2 * w_box), 0)
    x2 = min(x + int(0.8 * w_box), w)
    y1 = max(y, 0)
    y2 = min(y + int(0.25 * h_box), h)

    return x1, y1, x2, y2


# Buffer untuk menyimpan sinyal dan timestamp
# maxlen menentukan berapa banyak sampel yang disimpan (misal ~10 detik jika 30 fps)
signal_buffer = deque(maxlen=300)
time_buffer = deque(maxlen=300)


# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Estimasi FPS dari kamera, fallback ke 30 jika tidak tersedia
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30

print(f"Menggunakan FPS perkiraan: {fps}")


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera. Pastikan webcam terhubung.")
            break

        # Konversi ke RGB untuk MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            # Ambil deteksi wajah pertama
            detection = results.detections[0]
            x1, y1, x2, y2 = get_forehead_roi(frame, detection)

            # Pastikan ROI valid
            if y2 > y1 and x2 > x1:
                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    # Kanal hijau
                    green_channel = roi[:, :, 1]
                    mean_val = float(np.mean(green_channel))

                    # Simpan ke buffer
                    signal_buffer.append(mean_val)
                    time_buffer.append(time.time())

                    # Gambar kotak ROI di frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Tampilkan panjang buffer secara berkala
                    if len(signal_buffer) % 30 == 0:
                        print(f"Panjang buffer sinyal: {len(signal_buffer)} sampel")

        # Tampilkan video dengan ROI
        cv2.imshow("Webcam + ROI dahi", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Keluar dari program atas perintah pengguna.")
            break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()

# Setelah program berhenti, Anda bisa memeriksa isi buffer
print(f"Total sampel sinyal yang terekam: {len(signal_buffer)}")
