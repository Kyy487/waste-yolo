# Panduan Menjalankan Project

Panduan ini menjelaskan cara menjalankan bagian yang sudah tersedia di repository ini. Jalankan semua perintah dari root repository (`C:\waste-yolo`) dengan virtual environment aktif.

## Status modul

| Modul | Status | Perintah tersedia |
| --- | --- | --- |
| Pipeline data | Selesai | `python -m src.data.prepare_data ...` |
| Training | Selesai | `python -m src.train.train ...` |
| Evaluasi/test/ekspor ONNX | Belum diimplementasikan | - |
| API FastAPI | Selesai (kode + test) | `uvicorn src.api.main:app --reload --port 8000` |
| Demo Streamlit | Selesai (kode + test) | `streamlit run app/streamlit_app.py` |

`src/eval/evaluate.py` masih kosong. API dan demo sudah bisa dijalankan, tetapi prediksi nyata membutuhkan `models/model.onnx` yang belum diekspor. Tanpa model, `/v1/health` melaporkan `model_loaded: false` dan `/v1/predict` mengembalikan `503 MODEL_NOT_LOADED`.

## 1. Prasyarat

- Python 3.10 atau lebih baru. Lingkungan pengembangan memakai Python 3.12.
- Git.
- Dataset MWCD sudah diekstrak ke `data/raw/`.
- GPU opsional. Tanpa GPU, training berjalan di CPU dan jauh lebih lama.
- Koneksi internet saat pertama kali karena Ultralytics mengunduh `yolov8n-cls.pt` dan font Arial bila belum ada di cache.

Struktur data mentah yang diharapkan:

```text
data/raw/
├── battery/
├── cardboard/
├── glass/
├── metal/
├── organic/
├── paper/
├── plastic/
├── textile/
└── trash/
```

Nama folder harus persis 9 kelas di atas. Ekstensi yang diterima: `.jpg`, `.jpeg`, `.png`. Jangan meng-commit isi `data/raw/`.

Sitasi dataset: Israt Jahan, Sanjida Tasneem, Fahmida Rifha. *Merged Waste Classification Dataset (MWCD)*. Mendeley Data, V1. DOI: `10.17632/x863v66fv3.1`. Lisensi CC BY 4.0.

## 2. Instalasi

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Cek instalasi:

```bash
python -c "import torch, ultralytics, cv2, numpy, yaml; print(torch.__version__, ultralytics.__version__, cv2.__version__, numpy.__version__)"
```

## 3. Verifikasi repository

```bash
ruff check .
black --check .
pytest -q
```

Pada lingkungan ini, `pytest -q` menghasilkan **46 passed**. Warning depresasi `pyparsing` dari matplotlib dan `anyio` dari Starlette TestClient tidak menggagalkan test.

## 4. Menjalankan pipeline data

```bash
python -m src.data.prepare_data --config configs/data.yaml --root .
```

Perintah ini:

- membaca gambar dari `data/raw/`;
- membuang gambar korup;
- melakukan deduplikasi SHA-1 dan perceptual hash;
- membuang gambar dengan label konflik;
- membuat split stratified 70% train / 15% val / 15% test dengan seed `42`;
- menulis dataset YOLO-cls 224×224 ke `data/processed/dataset/`;
- menulis `data/processed/splits.csv` dan `data/processed/dedup_removed.csv`;
- menulis ringkasan ke `results/data_audit.csv` dan `results/class_distribution.png`.

Hasil eksekusi terakhir di repository ini: 35.168 gambar menjadi 30.644 gambar (train 21.450, val 4.597, test 4.597). Ulangi perintah hanya bila data mentah berubah, karena `prepare_data.py` menolak menimpa hasil yang sudah ada.

## 5. Menjalankan training

Semua hyperparameter dibaca dari [`configs/train.yaml`](../configs/train.yaml). Gunakan CLI override hanya untuk eksperimen yang sengaja berbeda, dan catat di `notes`/`run_id`.

### 5.1 Smoke test

Gunakan smoke test untuk memastikan training bisa berjalan tanpa mengisi tabel eksperimen:

```bash
python -m src.train.train --config configs/train.yaml --root . --run-id E0_smoke_ulang_yolov8n-cls_224 --epochs 1 --fraction 0.005 --batch 8 --no-record
```

Ganti `--run-id` dengan nama unik bila dijalankan lagi. Di mesin ini smoke test selesai dalam 1,31 menit. Metrik smoke tidak boleh dipakai sebagai hasil evaluasi model.

Catatan: Ultralytics tetap menjalankan validasi pada epoch terakhir meskipun `--no-val` diberikan, jadi `--no-val` hanya mengurangi validasi antar-epoch.

### 5.2 Baseline E1

Konfigurasi default adalah `yolov8n-cls`, `epochs: 50`, `imgsz: 224`, `batch: 32`, `patience: 10`, seed `42`.

```bash
python -m src.train.train --config configs/train.yaml --root .
```

Estimasi kasar di CPU yang sama: 9–10 jam. Sesuai `AGENTS.md`, training panjang ini memerlukan konfirmasi Project Lead sebelum dijalankan.

### 5.3 Contoh eksperimen pendek

```bash
python -m src.train.train --config configs/train.yaml --root . --run-id E2_yolov8n-cls_224_short --epochs 10 --fraction 0.1 --batch 16 --notes "eksperimen 10 epoch pada 10 persen data"
```

Ikuti pola nama `E{nomor}_{model}_{imgsz}_{catatan}` dan ubah satu variabel per eksperimen agar hasil dapat dibandingkan.

### Artefak training

| Path | Isi |
| --- | --- |
| `results/runs/<run_id>/config.yaml` | Salinan konfigurasi yang dipakai |
| `results/runs/<run_id>/train.log` | Log training |
| `results/runs/<run_id>/metadata.json` | commit hash, versi library, durasi, metrik, checkpoint |
| `runs/classify/<run_id>/weights/best.pt` | Checkpoint terbaik dari Ultralytics |
| `results/experiments.csv` | Ringkasan semua run, kecuali yang memakai `--no-record` |

File `.pt` dan `.onnx` sudah diabaikan oleh `.gitignore`. Jangan meng-commit bobot model atau dataset.

## 6. Menjalankan API FastAPI

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoint tersedia di `http://localhost:8000/v1`: `/health`, `/classes`, `/model`, `/predict`, `/predict/batch`. Dokumentasi otomatis ada di `http://localhost:8000/docs`.

API memuat `models/model.onnx` sekali saat startup. Bila file belum ada, layanan tetap hidup tetapi `/v1/predict` dan `/v1/predict/batch` mengembalikan `503 MODEL_NOT_LOADED`. Praproses, validasi upload, dan format error mengikuti [`api_contract.md`](api_contract.md).

Contoh prediksi satu gambar:

```bash
curl -X POST "http://localhost:8000/v1/predict?top_k=3" -F "file=@contoh.jpg"
```

## 7. Menjalankan demo kamera real-time

Jalankan API terlebih dahulu, lalu:

```bash
streamlit run app/streamlit_app.py
```

Demo memakai pola **satu frame per rerun** Streamlit (`st.rerun()`), bukan loop `while` yang memblokir. Tombol **Start kamera** membuka `cv2.VideoCapture`, mengirim frame JPEG in-memory ke `/v1/predict`, menampilkan overlay label/confidence/FPS, lalu me-rerun halaman. Tombol **Stop kamera** melepaskan kamera. Pengaturan kamera ada di bagian `demo:` pada `configs/api.yaml`.

Demo tidak memuat model sendiri dan tidak menyimpan frame ke disk. Prediksi nyata baru muncul setelah `models/model.onnx` tersedia.

## 8. Masalah umum

| Masalah | Penyebab | Solusi |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'src'` | Perintah dijalankan dari folder lain atau venv belum aktif | Jalankan dari `C:\waste-yolo` dan aktifkan `.venv` |
| `FileNotFoundError` pada `data/raw` | Dataset belum diekstrak | Letakkan 9 folder kelas di `data/raw/` |
| `FileExistsError: run directory already exists` | `--run-id` sudah dipakai | Gunakan run ID baru |
| `run_id already exists in ... experiments.csv` | Run ID sudah tercatat | Gunakan run ID baru atau `--no-record` untuk smoke |
| Training sangat lambat | CPU-only | Kecilkan `--batch` atau `--fraction`, atau gunakan GPU |
| `black` mengabaikan notebook | Extra Jupyter tidak terpasang | Aman untuk pipeline Python; notebook bukan sumber logika utama |
| `/v1/predict` mengembalikan `503 MODEL_NOT_LOADED` | `models/model.onnx` belum ada | Selesaikan training, evaluasi, dan ekspor ONNX terlebih dahulu |
| Demo menampilkan “API tidak tersedia” | Uvicorn belum berjalan atau port berbeda | Jalankan `uvicorn src.api.main:app --port 8000` dan cek `configs/api.yaml` |
| Kamera tidak dapat dibuka | `camera_index` salah atau kamera dipakai aplikasi lain | Ubah `demo.camera_index` di `configs/api.yaml` dan tutup aplikasi lain |

## 9. Tahap berikutnya

Setelah baseline dan pemilihan model selesai, tahap berikutnya adalah evaluasi test set yang terkunci, ekspor ONNX, API FastAPI `/v1`, dan demo Streamlit. Test set hanya dipakai sekali untuk evaluasi akhir model terpilih dan tidak boleh dipakai untuk tuning.

Aturan lengkap ada di [`AGENTS.md`](../AGENTS.md), rancangan umum di [`RANCANGAN.md`](RANCANGAN.md), dan rancangan kamera real-time di [`RANCANGAN_REALTIME.md`](RANCANGAN_REALTIME.md).
