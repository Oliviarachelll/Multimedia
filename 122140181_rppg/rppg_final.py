import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
from scipy import signal
import matplotlib.pyplot as plt

"""rPPG Final Script (dengan fitur tambahan)

File ini menggabungkan semua tahapan tugas rPPG dalam satu skrip Python.
Di dalamnya sudah ada kode + penjelasan sederhana tentang apa yang dilakukan.

Ringkasan alur sistem:
1. Baca video dari webcam.
2. Deteksi wajah dengan MediaPipe. Ambil ROI di dahi.
3. Hitung rata-rata kanal hijau di ROI sebagai sinyal rPPG mentah.
4. Simpan sinyal ke buffer dengan konsep sliding window (sekitar 10 detik).
5. Filter sinyal dengan bandpass Butterworth 0.7–4 Hz.
6. Hitung spektrum frekuensi dengan FFT, cari puncak di rentang BPM wajar.
7. Konversi frekuensi puncak menjadi BPM.
8. Stabilkan nilai BPM dengan exponential smoothing.
9. Deteksi gerakan sederhana dari perpindahan ROI (Motion: OK / HIGH).
10. Auto-scaling ROI sesuai ukuran wajah.
11. Tampilkan video + BPM + status gerakan + dua grafik (sinyal dan spektrum).

Fitur tambahan:
- Stabilizer BPM (exponential smoothing) supaya BPM tidak lompat-lompat.
- Motion detection sederhana untuk memberi tahu jika gerakan terlalu besar.
- Auto-scaling ROI dahi sesuai ukuran wajah (dekat/jauh dari kamera).
- Visualisasi sinyal rPPG dan spektrum dalam BPM secara real time.

Catatan:
- Gunakan skrip ini sebagai file utama pengumpulan (misalnya rppg_final.py).
- Pastikan semua library sudah terpasang sebelum menjalankan.
"""

# ======================
# Konfigurasi utama
# ======================

WINDOW_SECONDS = 10.0          # panjang jendela waktu untuk estimasi BPM (detik)
LOW_HZ = 0.7                   # batas bawah bandpass (Hz)
HIGH_HZ = 4.0                  # batas atas bandpass (Hz)
MIN_BPM = 40.0                 # batas bawah BPM yang dianggap wajar
MAX_BPM = 180.0                # batas atas BPM yang dianggap wajar
ALPHA_SMOOTH = 0.3             # koefisien smoothing BPM (0-1)
MOTION_THRESH_REL = 0.05       # ambang gerakan relatif ROI

mp_face_detection = mp.solutions.face_detection


def get_forehead_roi(frame: np.ndarray, detection: mp_face_detection.FaceDetection) -> tuple:
    """Menghitung koordinat ROI dahi dari bounding box wajah.

    ROI otomatis diskalakan berdasarkan ukuran wajah.
    - Jika wajah sangat dekat ke kamera, ROI dibuat sedikit lebih sempit.
    - Jika wajah cukup jauh, ROI dibuat sedikit lebih lebar.
    Tujuannya agar area dahi yang diambil tetap proporsional.
    """
    h, w, _ = frame.shape
    bboxC = detection.location_data.relative_bounding_box

    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    w_box = int(bboxC.width * w)
    h_box = int(bboxC.height * h)

    # Ukuran relatif tinggi wajah terhadap tinggi frame
    face_rel = h_box / max(h, 1)

    # Penyesuaian dinamis ROI berdasarkan ukuran wajah
    if face_rel > 0.4:  # wajah sangat dekat
        x_left_factor = 0.3
        x_right_factor = 0.7
        forehead_height_factor = 0.18
    elif face_rel < 0.2:  # wajah agak jauh
        x_left_factor = 0.15
        x_right_factor = 0.85
        forehead_height_factor = 0.30
    else:  # normal
        x_left_factor = 0.2
        x_right_factor = 0.8
        forehead_height_factor = 0.25

    x1 = max(x + int(x_left_factor * w_box), 0)
    x2 = min(x + int(x_right_factor * w_box), w)
    y1 = max(y, 0)
    y2 = min(y + int(forehead_height_factor * h_box), h)

    return x1, y1, x2, y2


"""Penjelasan singkat get_forehead_roi:
Fungsi ini hanya mengubah bounding box wajah menjadi ROI dahi.
ROI dibuat dinamis (auto-scaling) mengikuti besar kecilnya wajah.
Hal ini membuat area dahi tetap proporsional ketika jarak pengguna ke kamera berubah.
"""


def bandpass_filter(signal_in: np.ndarray, fs: float,
                    low: float = LOW_HZ, high: float = HIGH_HZ,
                    order: int = 4) -> np.ndarray:
    """Menerapkan bandpass Butterworth ke sinyal 1D.

    - low, high dalam satuan Hz.
    - fs adalah sampling rate (FPS kamera).
    Sinyal di luar rentang [low, high] akan dilemahkan.
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    b, a = signal.butter(order, [low_norm, high_norm], btype="bandpass")
    filtered = signal.filtfilt(b, a, signal_in)
    return filtered


"""Penjelasan singkat bandpass_filter:
Fungsi ini membersihkan sinyal rPPG dengan hanya mempertahankan
frekuensi yang masuk akal untuk detak jantung manusia.
"""


def estimate_bpm_fft(signal_in: np.ndarray, fs: float,
                     min_bpm: float = MIN_BPM,
                     max_bpm: float = MAX_BPM):
    """Menghitung BPM dari sinyal menggunakan FFT.

    Mengembalikan tiga nilai:
    - bpm_est  : estimasi BPM (float). 0 jika tidak valid.
    - bpm_axis : sumbu BPM untuk spektrum.
    - spectrum : magnituda FFT di setiap BPM.
    """
    if len(signal_in) < 30:
        return 0.0, None, None

    # Hilangkan mean agar sinyal berpusat di sekitar nol
    sig = signal_in - np.mean(signal_in)
    n = len(sig)

    # FFT satu sisi (real input)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(sig))

    # Konversi BPM ke Hz
    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0

    # Ambil frekuensi di rentang tersebut
    valid_idx = np.where((freqs >= min_hz) & (freqs <= max_hz))
    if len(valid_idx[0]) == 0:
        return 0.0, None, None

    freqs_valid = freqs[valid_idx]
    fft_valid = fft_vals[valid_idx]

    # Ambil frekuensi dengan amplitudo terbesar
    peak_idx = np.argmax(fft_valid)
    peak_freq = freqs_valid[peak_idx]
    bpm = float(peak_freq * 60.0)

    bpm_axis = freqs_valid * 60.0
    spectrum = fft_valid
    return bpm, bpm_axis, spectrum


"""Penjelasan singkat estimate_bpm_fft:
Fungsi ini mengubah sinyal waktu menjadi spektrum frekuensi.
Puncak tertinggi di rentang 40–180 BPM dipilih sebagai detak jantung.
Nilai bpm_axis dan spectrum dipakai untuk plot spektrum.
"""


def init_plots():
    """Menyiapkan dua grafik matplotlib:

    - Grafik atas : sinyal rPPG terfilter terhadap waktu.
    - Grafik bawah: spektrum rPPG dalam satuan BPM.
    """
    plt.ion()  # mode interaktif
    fig, (ax_sig, ax_fft) = plt.subplots(2, 1, figsize=(6, 6))

    # Grafik sinyal waktu
    line_sig, = ax_sig.plot([], [])
    ax_sig.set_title("Sinyal rPPG terfilter")
    ax_sig.set_xlabel("Waktu (detik)")
    ax_sig.set_ylabel("Intensitas relatif")

    # Grafik spektrum BPM
    line_fft, = ax_fft.plot([], [])
    ax_fft.set_title("Spektrum rPPG (BPM)")
    ax_fft.set_xlabel("BPM")
    ax_fft.set_ylabel("Magnituda")
    ax_fft.set_xlim(MIN_BPM, MAX_BPM)

    fig.tight_layout()
    return fig, ax_sig, ax_fft, line_sig, line_fft


def update_plots(fig, ax_sig, ax_fft, line_sig, line_fft,
                  sig_filtered: np.ndarray,
                  fps: float,
                  bpm_axis: np.ndarray | None,
                  spectrum: np.ndarray | None) -> None:
    """Memperbarui isi dua grafik secara real time.

    - sig_filtered: sinyal rPPG yang sudah difilter.
    - fps        : frame rate kamera.
    - bpm_axis   : sumbu BPM.
    - spectrum   : magnituda FFT.
    """
    if sig_filtered is None or len(sig_filtered) == 0:
        return

    # Grafik sinyal waktu
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

    # Grafik spektrum BPM
    if bpm_axis is not None and spectrum is not None and len(bpm_axis) > 0:
        line_fft.set_data(bpm_axis, spectrum)
        ax_fft.set_xlim(MIN_BPM, MAX_BPM)

        smin = float(np.min(spectrum))
        smax = float(np.max(spectrum))
        if smin == smax:
            smin = 0.0
            smax += 1.0
        ax_fft.set_ylim(smin, smax)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


"""Penjelasan singkat init_plots dan update_plots:
Dua fungsi ini mengurus tampilan grafik.
- init_plots: membuat dua subplot kosong.
- update_plots: mengisi data sinyal dan spektrum setiap kali loop.
Visualisasi ini membantu melihat kualitas sinyal dan posisi puncak frekuensi.
"""


def main() -> None:
    """Fungsi utama untuk menjalankan rPPG secara real time.

    Langkah besar:
    1. Buka webcam dan baca FPS.
    2. Deteksi wajah dan ROI dahi (dengan auto-scaling sesuai ukuran wajah).
    3. Kumpulkan sinyal rPPG mentah di buffer (sliding window 10 detik).
    4. Filter sinyal dan hitung BPM dengan FFT.
    5. Stabilkan BPM dengan exponential smoothing.
    6. Deteksi gerakan wajah sederhana dari perpindahan ROI.
    7. Tampilkan video dan dua grafik yang terus diperbarui.
    """
    # Buffer sinyal dan waktu
    signal_buffer = deque()
    time_buffer = deque()

    # Variabel untuk smoothing BPM dan deteksi gerakan
    bpm_smooth = 0.0
    prev_cx = None
    prev_cy = None
    motion_flag = "OK"

    # Buka kamera
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    print(f"Menggunakan FPS perkiraan: {fps}")

    # Inisialisasi grafik matplotlib
    fig, ax_sig, ax_fft, line_sig, line_fft = init_plots()

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5,
    ) as face_detection:
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
                        # Deteksi gerakan sederhana dari perpindahan pusat ROI
                        h, w, _ = frame.shape
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        if prev_cx is not None and prev_cy is not None:
                            dx = (cx - prev_cx) / max(w, 1)
                            dy = (cy - prev_cy) / max(h, 1)
                            motion_mag = float((dx * dx + dy * dy) ** 0.5)
                            if motion_mag > MOTION_THRESH_REL:
                                motion_flag = "HIGH"
                            else:
                                motion_flag = "OK"
                        prev_cx, prev_cy = cx, cy

                        # Ambil rata-rata kanal hijau sebagai sinyal rPPG mentah
                        green_channel = roi[:, :, 1]
                        mean_val = float(np.mean(green_channel))
                        t_now = time.time()

                        # Simpan ke buffer
                        signal_buffer.append(mean_val)
                        time_buffer.append(t_now)

                        # Hapus sampel yang lebih tua dari WINDOW_SECONDS
                        while time_buffer and (t_now - time_buffer[0]) > WINDOW_SECONDS:
                            time_buffer.popleft()
                            signal_buffer.popleft()

                        # Ubah buffer ke numpy array
                        sig_arr = np.array(signal_buffer, dtype=np.float32)

                        # Minimal butuh beberapa detik data untuk estimasi BPM
                        if len(sig_arr) > fps * 3:
                            try:
                                # Filtering bandpass
                                sig_filtered = bandpass_filter(sig_arr, fs=fps)

                                # Estimasi BPM dengan FFT
                                bpm_est, bpm_axis, spectrum = estimate_bpm_fft(
                                    sig_filtered, fs=fps
                                )
                            except Exception as e:  # jaga-jaga kalau filter/FFT error
                                print(f"Terjadi error saat filtering/FFT: {e}")
                                bpm_est = 0.0

                        # Gambar ROI pada frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Stabilizer BPM dengan exponential smoothing
            if bpm_est > 0.0:
                if bpm_smooth <= 0.0:
                    bpm_smooth = bpm_est
                else:
                    bpm_smooth = (
                        ALPHA_SMOOTH * bpm_est
                        + (1.0 - ALPHA_SMOOTH) * bpm_smooth
                    )

            # Pilih BPM yang ditampilkan
            if bpm_smooth > 0.0:
                text_bpm = f"BPM: {bpm_smooth:.1f}"
            elif bpm_est > 0.0:
                text_bpm = f"BPM: {bpm_est:.1f}"
            else:
                text_bpm = "BPM: --"

            # Teks BPM
            cv2.putText(
                frame,
                text_bpm,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            # Teks status gerakan
            cv2.putText(
                frame,
                f"Motion: {motion_flag}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255) if motion_flag == "HIGH" else (0, 255, 0),
                2,
            )

            # Tampilkan video
            cv2.imshow("rPPG real-time (final)", frame)

            # Perbarui grafik sinyal dan spektrum
            update_plots(
                fig,
                ax_sig,
                ax_fft,
                line_sig,
                line_fft,
                sig_filtered,
                fps,
                bpm_axis,
                spectrum,
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Keluar atas perintah pengguna.")
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)


"""Penjelasan singkat main:
Fungsi main menjalankan seluruh pipeline rPPG secara real time.
Di dalam loop:
- Baca frame dari webcam.
- Deteksi wajah dan hitung ROI dahi.
- Deteksi gerakan dari perpindahan ROI.
- Ambil sinyal hijau, simpan di buffer sliding window.
- Filter sinyal dan hitung BPM dengan FFT.
- Stabilkan BPM dengan exponential smoothing.
- Tampilkan video + BPM + status gerakan.
- Perbarui grafik sinyal dan spektrum.
"""


if __name__ == "__main__":
    # Sebelum menjalankan, pastikan paket berikut sudah terpasang:
    #   pip install opencv-python mediapipe numpy scipy matplotlib
    main()
