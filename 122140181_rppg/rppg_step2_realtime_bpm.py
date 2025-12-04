import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from scipy import signal

"""
Step 2: Ekstraksi sinyal rPPG + filtering + estimasi BPM secara real-time.

File ini berdiri sendiri, tapi struktur dasarnya mirip dengan rppg_step1_roi_signal.py.
Perbedaan utama:
- Masih ambil sinyal kanal hijau dari ROI dahi.
- Tambahan: detrending dan bandpass filter 0.7-4 Hz.
- Menghitung BPM dengan FFT di jendela waktu (sliding window).
- Menampilkan BPM di jendela video.

Tekan 'q' untuk keluar.

Catatan:
- Pastikan scipy sudah terinstall: `pip install scipy`.
- FPS diestimasi dari kamera. Jika salah jauh, BPM bisa meleset.
"""

mp_face_detection = mp.solutions.face_detection


def get_forehead_roi(frame: np.ndarray, detection: mp_face_detection.FaceDetection) -> tuple:
    """Hitung koordinat ROI dahi berdasarkan bounding box wajah.

    Mengembalikan (x1, y1, x2, y2).
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


def bandpass_filter(signal_in: np.ndarray, fs: float, low: float = 0.7, high: float = 4.0, order: int = 4) -> np.ndarray:
    """Terapkan Butterworth bandpass filter ke sinyal.

    low dan high dalam satuan Hz.
    fs adalah sampling rate (FPS kamera).
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    b, a = signal.butter(order, [low_norm, high_norm], btype="bandpass")
    filtered = signal.filtfilt(b, a, signal_in)
    return filtered


def estimate_bpm_fft(signal_in: np.ndarray, fs: float, min_bpm: float = 40.0, max_bpm: float = 180.0) -> float:
    """Estimasi BPM dengan FFT dari sinyal 1D.

    Mengembalikan nilai BPM dominan di rentang [min_bpm, max_bpm].
    Jika gagal, mengembalikan 0.
    """
    if len(signal_in) < 30:
        return 0.0

    # Hilangkan mean
    sig = signal_in - np.mean(signal_in)

    # FFT
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(sig))

    # Konversi rentang BPM ke Hz
    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0

    # Ambil indeks di rentang tersebut
    valid_idx = np.where((freqs >= min_hz) & (freqs <= max_hz))
    if len(valid_idx[0]) == 0:
        return 0.0

    freqs_valid = freqs[valid_idx]
    fft_valid = fft_vals[valid_idx]

    # Frekuensi dengan amplitudo maksimum
    peak_idx = np.argmax(fft_valid)
    peak_freq = freqs_valid[peak_idx]

    bpm = float(peak_freq * 60.0)
    return bpm


# Buffer sinyal dan timestamp untuk sliding window
WINDOW_SECONDS = 10  # panjang jendela waktu untuk estimasi BPM
signal_buffer = deque()
time_buffer = deque()


cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0

print(f"Menggunakan FPS perkiraan: {fps}")


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        bpm_est = 0.0

        if results.detections:
            detection = results.detections[0]
            x1, y1, x2, y2 = get_forehead_roi(frame, detection)

            if y2 > y1 and x2 > x1:
                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    green_channel = roi[:, :, 1]
                    mean_val = float(np.mean(green_channel))
                    t_now = time.time()

                    signal_buffer.append(mean_val)
                    time_buffer.append(t_now)

                    # Hapus data lama di luar window
                    while time_buffer and (t_now - time_buffer[0]) > WINDOW_SECONDS:
                        time_buffer.popleft()
                        signal_buffer.popleft()

                    # Konversi ke numpy array untuk pemrosesan
                    sig_arr = np.array(signal_buffer, dtype=np.float32)

                    if len(sig_arr) > fps * 3:  # minimal ~3 detik data
                        try:
                            # Bandpass filter
                            sig_filtered = bandpass_filter(sig_arr, fs=fps)

                            # Estimasi BPM dengan FFT
                            bpm_est = estimate_bpm_fft(sig_filtered, fs=fps)
                        except Exception as e:
                            # Jika filter error, abaikan dan lanjut
                            # print(f"Error filter: {e}")
                            bpm_est = 0.0

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Tampilkan BPM di frame
        if bpm_est > 0:
            text = f"BPM: {bpm_est:.1f}"
        else:
            text = "BPM: --"

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("rPPG real-time (Step 2)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Keluar atas perintah pengguna.")
            break

cap.release()
cv2.destroyAllWindows()
