import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from scipy import signal
import matplotlib.pyplot as plt

"""
Step 3: Visualisasi sinyal rPPG dan spektrum FFT secara real-time.

Fitur utama:
- Tetap mendeteksi wajah dan mengambil ROI dahi.
- Mengambil sinyal kanal hijau dan menyimpan ke buffer sliding window.
- Melakukan bandpass filter 0.7-4 Hz.
- Menghitung BPM dengan FFT seperti Step 2.
- Menampilkan:
  1) Video dengan ROI dan teks BPM.
  2) Grafik sinyal rPPG terfilter vs waktu.
  3) Grafik spektrum frekuensi (dalam BPM).

Tekan 'q' di jendela video untuk keluar.
Pastikan scipy dan matplotlib sudah terpasang.
"""

mp_face_detection = mp.solutions.face_detection


def get_forehead_roi(frame: np.ndarray, detection: mp_face_detection.FaceDetection) -> tuple:
    """Hitung koordinat ROI dahi berdasarkan bounding box wajah."""
    h, w, _ = frame.shape
    bboxC = detection.location_data.relative_bounding_box

    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    w_box = int(bboxC.width * w)
    h_box = int(bboxC.height * h)

    x1 = max(x + int(0.2 * w_box), 0)
    x2 = min(x + int(0.8 * w_box), w)
    y1 = max(y, 0)
    y2 = min(y + int(0.25 * h_box), h)

    return x1, y1, x2, y2


def bandpass_filter(signal_in: np.ndarray, fs: float, low: float = 0.7, high: float = 4.0, order: int = 4) -> np.ndarray:
    """Terapkan bandpass Butterworth ke sinyal satu dimensi."""
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    b, a = signal.butter(order, [low_norm, high_norm], btype="bandpass")
    filtered = signal.filtfilt(b, a, signal_in)
    return filtered


def estimate_bpm_fft(signal_in: np.ndarray, fs: float, min_bpm: float = 40.0, max_bpm: float = 180.0):
    """Estimasi BPM dan juga mengembalikan sumbu dan spektrum untuk plotting."""
    if len(signal_in) < 30:
        return 0.0, None, None

    sig = signal_in - np.mean(signal_in)
    n = len(sig)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(sig))

    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0

    valid_idx = np.where((freqs >= min_hz) & (freqs <= max_hz))
    if len(valid_idx[0]) == 0:
        return 0.0, None, None

    freqs_valid = freqs[valid_idx]
    fft_valid = fft_vals[valid_idx]

    peak_idx = np.argmax(fft_valid)
    peak_freq = freqs_valid[peak_idx]
    bpm = float(peak_freq * 60.0)

    # Sumbu BPM dan spektrum untuk plot
    bpm_axis = freqs_valid * 60.0
    spectrum = fft_valid
    return bpm, bpm_axis, spectrum


WINDOW_SECONDS = 10
signal_buffer = deque()
time_buffer = deque()


cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0

print(f"Menggunakan FPS perkiraan: {fps}")

# Siapkan plot matplotlib
plt.ion()
fig, (ax_sig, ax_fft) = plt.subplots(2, 1, figsize=(6, 6))

line_sig, = ax_sig.plot([], [])
ax_sig.set_title("Sinyal rPPG terfilter")
ax_sig.set_xlabel("Waktu (detik)")
ax_sig.set_ylabel("Intensitas relatif")

line_fft, = ax_fft.plot([], [])
ax_fft.set_title("Spektrum rPPG (BPM)")
ax_fft.set_xlabel("BPM")
ax_fft.set_ylabel("Magnituda")
ax_fft.set_xlim(40, 180)

fig.tight_layout()


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        bpm_est = 0.0
        bpm_axis = None
        spectrum = None
        sig_filtered = None

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

                    while time_buffer and (t_now - time_buffer[0]) > WINDOW_SECONDS:
                        time_buffer.popleft()
                        signal_buffer.popleft()

                    sig_arr = np.array(signal_buffer, dtype=np.float32)

                    if len(sig_arr) > fps * 3:
                        try:
                            sig_filtered = bandpass_filter(sig_arr, fs=fps)
                            bpm_est, bpm_axis, spectrum = estimate_bpm_fft(sig_filtered, fs=fps)
                        except Exception:
                            bpm_est = 0.0
                            sig_filtered = None
                            bpm_axis = None
                            spectrum = None

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if bpm_est > 0:
            text = f"BPM: {bpm_est:.1f}"
        else:
            text = "BPM: --"

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("rPPG real-time dengan visualisasi (Step 3)", frame)

        # Update plot jika ada sinyal terfilter
        if sig_filtered is not None and len(sig_filtered) > 0:
            n = len(sig_filtered)
            t_axis = np.linspace(0, n / fps, n)

            line_sig.set_data(t_axis, sig_filtered)
            ax_sig.set_xlim(0, max(t_axis))
            ymin = float(np.min(sig_filtered))
            ymax = float(np.max(sig_filtered))
            if ymin == ymax:
                ymin -= 1.0
                ymax += 1.0
            ax_sig.set_ylim(ymin, ymax)

            if bpm_axis is not None and spectrum is not None:
                line_fft.set_data(bpm_axis, spectrum)
                ax_fft.set_xlim(40, 180)
                smin = float(np.min(spectrum))
                smax = float(np.max(spectrum))
                if smin == smax:
                    smin = 0.0
                    smax += 1.0
                ax_fft.set_ylim(smin, smax)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Keluar atas perintah pengguna.")
            break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close(fig)
