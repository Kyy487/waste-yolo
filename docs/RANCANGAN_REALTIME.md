# Rancangan Klasifikasi Sampah Real-Time via Kamera

Status: API dan demo sudah diimplementasikan; menunggu `models/model.onnx` dan persetujuan Project Lead  
Tanggal: 2026-11-23  
Keputusan mode: **klasifikasi real-time**, bukan deteksi dengan bounding box.

## 1. Latar belakang

Dataset MWCD hanya memiliki label per gambar, tanpa koordinat bounding box. Sesuai `AGENTS.md`, project ini adalah klasifikasi 9 kelas, bukan object detection. Pengguna menginginkan demo dengan kamera real-time; oleh karena itu demo akan menampilkan **satu label kelas per frame**, bukan beberapa kotak objek.

Pipeline data, modul training, API FastAPI, dan demo Streamlit sudah tersedia di sisi kode. Evaluasi final dan ekspor ONNX belum dilakukan, sehingga `models/model.onnx` belum ada. Kamera sudah bisa mengirim frame ke API, tetapi prediksi nyata baru muncul setelah artefak ONNX siap.

Sumber kebenaran kontrak: `AGENTS.md` bagian 8 dan 9.

## 2. Tujuan

1. Menampilkan prediksi kelas sampah dari webcam secara langsung dengan label dan confidence.
2. Memastikan demo hanya memanggil API, tidak memuat model sendiri.
3. Menjaga praproses tetap identik dengan training: resize 224×224 dengan padding, RGB.
4. Menyediakan perilaku error yang jelas saat kamera atau API bermasalah.
5. Tidak menyimpan gambar pengguna dan tidak memakai layanan AI pihak ketiga.

## 3. Di luar cakupan

- Bounding box, multiple object detection, dan tracking antar-frame.
- Pengenalan individu, wajah, atau data pribadi.
- Penyimpanan frame ke disk secara default.
- Deployment publik dengan autentikasi, autoscaling, atau monitoring penuh.
- Perubahan urutan kelas atau skema respons API.

## 4. Arsitektur

```text
[Webcam]
   |  cv2.VideoCapture, frame BGR
   v
[Klien demo / Streamlit]
   |  encode JPEG in-memory, POST /v1/predict?top_k=3
   v
[FastAPI v1]
   |  validasi file, decode, resize+padding 224, RGB
   v
[ONNX Runtime]
   |  output 9 probabilitas/logit
   v
[FastAPI v1]
   |  JSON: predicted_class, confidence, top_k, inference_ms, model_version
   v
[Klien demo / Streamlit]
   |  smoothing ringan + overlay teks
   v
[Tampilan real-time]
```

Alur ini menjaga batas modul: API memiliki model dan praproses; demo hanya mengirim frame dan merender hasil.

## 5. Komponen

### 5.1 Artefak model

- `models/model.onnx`: input `1×3×224×224`, output 9 nilai untuk kelas tetap.
- `models/model_card.json`: nama, versi, arsitektur, input size, daftar kelas, dataset, seed, commit, metrik nyata, keterbatasan.
- Urutan kelas tetap mengikuti `AGENTS.md` bagian 8.3:
  `battery`, `cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`, `textile`, `trash`.
- Test set terkunci dan hanya dipakai sekali untuk evaluasi akhir model terpilih.

### 5.2 API FastAPI

Endpoint wajib sesuai kontrak `v1`:

| Method | Path | Fungsi |
| --- | --- | --- |
| `GET` | `/v1/health` | Status layanan dan model termuat |
| `GET` | `/v1/classes` | 9 kelas beserta indeks |
| `GET` | `/v1/model` | Metadata model aktif |
| `POST` | `/v1/predict` | Prediksi satu frame/gambar |
| `POST` | `/v1/predict/batch` | Prediksi maksimal 16 gambar |

Perilaku penting:

- Model ONNX dimuat sekali saat startup, bukan per request.
- `top_k` default `3`, rentang `1–9`.
- Upload maksimal 5 MB, hanya JPG/PNG.
- Probabilitas di-normalisasi sehingga total seluruh kelas ≈ 1.
- Error memakai bentuk `{"error": {"code": ..., "message": ..., "details": ...}}`.
- Log request berisi waktu, endpoint, status, dan durasi tanpa menyimpan gambar.
- Aktifkan CORS hanya untuk origin lokal Streamlit bila diperlukan; ini tidak mengubah kontrak JSON.
- Semua nilai dibaca dari `configs/api.yaml`.

### 5.3 Klien kamera di Streamlit

File `app/streamlit_app.py` sudah diisi dengan perilaku:

1. Tombol **Start kamera** membuka kamera lewat `cv2.VideoCapture(camera_index)` dan menyimpannya di `st.session_state`.
2. Setiap rerun Streamlit mengambil **satu frame**, meng-encode ke JPEG di memory dengan `cv2.imencode`, lalu mengirimnya ke API.
3. Frame dikirim memakai satu `requests.Session` yang dipakai ulang.
4. Hanya satu request aktif per rerun; setelah respons diterima, halaman di-rerun dengan `st.rerun()`. Tidak ada loop `while` yang memblokir UI.
5. Respons API diparse, lalu label, confidence, dan FPS digambar sebagai overlay; `top_k` ditampilkan bila `show_top_k: true`.
6. Tombol **Stop kamera** melepaskan kamera dan menghapus state smoothing.
7. Frame tidak ditulis ke disk.

Klien **tidak** melakukan resize 224×224. Praproses tetap menjadi tanggung jawab API agar tidak terjadi normalisasi atau resize ganda.

### 5.4 Smoothing di sisi demo

Agar label tidak berkedip, demo melakukan majority vote pada `predicted_class` dari `smoothing_window` prediksi terakhir. Confidence yang ditampilkan adalah confidence dari prediksi terbaru, bukan rata-rata. Smoothing hanya berada di demo dan tidak mengubah kontrak API.

Nilai default yang diusulkan: `smoothing_window: 5`.

### 5.5 Konfigurasi tambahan

Bagian baru di `configs/api.yaml`, tanpa hard-code di kode:

```yaml
demo:
  camera_index: 0
  target_fps: 8
  top_k: 3
  smoothing_window: 5
  show_top_k: true
  request_timeout_s: 2.0
```

Base URL tidak ditulis ulang; diturunkan dari `api.host`, `api.port`, dan `api.prefix` yang sudah ada.

### 5.6 Dependency

| Package | Alasan |
| --- | --- |
| `requests` | Mengirim frame multipart dari Streamlit ke FastAPI |
| `httpx` | Dibutuhkan `TestClient` FastAPI untuk test kontrak |

Keduanya kecil dan harus dicatat di `requirements.txt` serta disebutkan di deskripsi PR.

## 6. Anggaran performa

- Target inferensi ONNX CPU dari `AGENTS.md`: **< 200 ms**. Ini target, bukan hasil terukur.
- Target tampilan demo: 5–10 FPS, diatur lewat `target_fps`.
- `request_timeout_s: 2.0` mencegah UI macet saat API lambat.
- Jika roundtrip lebih lambat dari target, demo melewati frame berikutnya daripada menumpuk request.
- Kecepatan aktual harus diukur dan dicatat di `results/CHANGELOG.md`; jangan mengisi angka perkiraan sebagai hasil.

## 7. Penanganan error

| Kondisi | Sumber | Perilaku demo |
| --- | --- | --- |
| Kamera tidak bisa dibuka | `cv2.VideoCapture` | Tampilkan pesan, tombol Start tetap tersedia |
| Frame kosong | OpenCV | Lewati frame, lanjutkan loop |
| API mati / timeout | Klien | Tampilkan status “API tidak tersedia”, frame tetap tampil tanpa prediksi baru |
| `503 MODEL_NOT_LOADED` | API | Tampilkan status “model belum siap” |
| `400 INVALID_IMAGE` | API | Tampilkan kode dan pesan error |
| `413 FILE_TOO_LARGE` | API | Tampilkan kode dan pesan error |
| `top_k` tidak valid | API | Tampilkan `INVALID_PARAMETER` |
| Tombol Stop | Pengguna | Lepaskan kamera, tutup session, hentikan loop |

Pesan error tidak boleh membocorkan path server atau stack trace.

## 8. Rencana file

| File | Aksi | Tujuan |
| --- | --- | --- |
| `src/api/main.py` | baru | FastAPI app, startup model load, 5 endpoint |
| `src/api/schemas.py` | baru | Model Pydantic sesuai `AGENTS.md` 8.6 |
| `src/api/inference.py` | baru | Session ONNX, preprocessing, softmax, top-k |
| `tests/test_api_contract.py` | baru | Test kontrak minimal sesuai `AGENTS.md` 11 |
| `app/api_client.py` | baru | Encode frame, kirim request, parse error |
| `app/streamlit_app.py` | diisi | UI kamera dan overlay real-time |
| `tests/test_camera_client.py` | baru | Test encoder dan mapping error dengan API di-mock |
| `configs/api.yaml` | diubah | Tambah bagian `demo:` |
| `requirements.txt` | diubah | Tambah `requests` dan `httpx` |
| `docs/api_contract.md` | baru | Referensi kontrak; bukan perubahan skema |
| `README.md` | diubah | Perintah uvicorn + streamlit dan status demo |
| `docs/PANDUAN_MENJALANKAN.md` | diubah | Langkah menjalankan demo kamera |
| `docs/RANCANGAN_REALTIME.md` | baru | Dokumen ini |

## 9. Tahap implementasi

Status kode: Tahap 2 (API) dan Tahap 3 (demo kamera) sudah diimplementasikan dan lolos test. Tahap 1 (model final + ONNX) dan Tahap 4 (video demo) belum selesai.

### Tahap 1 — Finalisasi model

Prasyarat sebelum API:

- Evaluasi akhir test set dilakukan satu kali oleh Evaluator.
- Model final dipilih manusia berdasarkan macro-F1 validation dan uji dunia nyata, bukan test set.
- `models/model.onnx` dan `models/model_card.json` tersedia.

**Acceptance:** ONNX dapat dimuat, output 9 kelas sesuai urutan, metrik di model card berasal dari eksekusi nyata.

### Tahap 2 — Implementasi API

- Buat skema Pydantic dan endpoint sesuai kontrak.
- Terapkan validasi upload, error seragam, logging, dan CORS lokal.
- Tulis `tests/test_api_contract.py`.

**Acceptance:** `pytest -q` lolos; `uvicorn src.api.main:app --reload --port 8000` berjalan; `/health` dan `/classes` sesuai kontrak.

### Tahap 3 — Implementasi demo kamera

- Buat klien API dan UI Streamlit.
- Terapkan throttle, smoothing, overlay, Start/Stop, dan penanganan error.
- Tulis `tests/test_camera_client.py` dengan request API di-mock.

**Acceptance:** kamera bisa dibuka, prediksi tampil, Stop melepaskan kamera, tidak ada frame yang disimpan.

### Tahap 4 — Dokumentasi dan video

- Perbarui README dan panduan menjalankan.
- Rekam video demo 1–2 menit yang menampilkan API + kamera.
- Catat metrik dan kendala nyata di `results/CHANGELOG.md`.

**Acceptance:** perintah di dokumentasi berhasil dijalankan dari repository bersih sesuai README.

## 10. Rencana pengujian

Test otomatis sudah ditulis dan dijalankan: `pytest -q` menghasilkan **46 passed**. Test kontrak API memakai engine tiruan karena `models/model.onnx` belum ada; checklist manual kamera tetap harus dijalankan setelah model tersedia.

### Test kontrak API

- `/health` → 200 dan `model_loaded: true`.
- `/classes` → 9 kelas dengan urutan benar.
- `/predict` dengan gambar valid → struktur sesuai `AGENTS.md` 8.3.
- File bukan gambar → 400 `INVALID_IMAGE`.
- File > 5 MB → 413 `FILE_TOO_LARGE`.
- Format selain JPG/PNG → 415 `UNSUPPORTED_MEDIA_TYPE`.
- Field wajib hilang → 422 `VALIDATION_ERROR`.
- `top_k` di luar 1–9 → 400 `INVALID_PARAMETER`.

### Test klien

- Frame valid menghasilkan JPEG bytes dengan nama `frame.jpg`.
- Frame kosong atau `None` ditolak.
- Respons sukses diparse menjadi label, confidence, dan daftar top-k.
- Error API dipetakan ke pesan UI yang aman.
- Tidak ada file gambar yang dibuat saat test.

### Checklist manual

- Kamera terbuka dan tertutup bersih.
- 30 frame berturut-turut menghasilkan label yang stabil.
- FPS dan `inference_ms` tercatat, bukan diasumsikan.
- Saat API dimatikan, UI tidak crash.
- Tidak ada wajah atau data pribadi yang sengaja direkam.
- Frame tidak tersimpan di `data/` maupun folder lain.

## 11. Risiko dan mitigasi

| Risiko | Mitigasi |
| --- | --- |
| Akurasi dunia nyata lebih rendah dari dataset | Evaluasi `data/realworld/` terpisah, laporkan jujur, jangan pakai test set untuk tuning |
| Label berkedip | Majority vote 5 frame terakhir |
| CPU terlalu lambat | Turunkan `target_fps`, ukur dulu sebelum mengubah model |
| Model final belum dipilih | Tidak mulai demo sebelum evaluasi dan persetujuan Lead |
| Contract drift | Perubahan kontrak wajib memperbarui `AGENTS.md`, `docs/api_contract.md`, skema, dan test; perubahan merusak memakai `/v2` |
| Dependency bertambah | Catat di `requirements.txt` dan jelaskan alasannya di PR |

## 12. Keputusan yang dibutuhkan

1. Setujui dokumen rancangan ini sebagai acuan implementasi.
2. Setujui penambahan dependency `requests` dan `httpx`.
3. Setujui penambahan bagian `demo:` di `configs/api.yaml`.
4. Konfirmasi model final setelah evaluasi selesai.
5. Konfirmasi training panjang bila baseline belum dijalankan.

## 13. Sitasi dataset

Israt Jahan, Sanjida Tasneem, Fahmida Rifha. *Merged Waste Classification Dataset (MWCD)*. Mendeley Data, V1. DOI: `10.17632/x863v66fv3.1`. Lisensi CC BY 4.0.
