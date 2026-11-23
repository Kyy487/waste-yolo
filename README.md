# Waste Classification with YOLO-cls (MWCD)

Klasifikasi 9 kelas sampah dari gambar menggunakan YOLO mode classify yang dilatih dengan transfer learning pada Merged Waste Classification Dataset (MWCD). Praproses memakai OpenCV. Proyek ini bukan object detection dan tidak memakai layanan AI pihak ketiga.

Panduan menjalankan: [`docs/PANDUAN_MENJALANKAN.md`](docs/PANDUAN_MENJALANKAN.md). Dokumen rancangan: [`docs/RANCANGAN.md`](docs/RANCANGAN.md). Rancangan kamera real-time: [`docs/RANCANGAN_REALTIME.md`](docs/RANCANGAN_REALTIME.md). Kontrak API: [`docs/api_contract.md`](docs/api_contract.md). Handoff tim: [`docs/HANDOFF.md`](docs/HANDOFF.md). Kontrak dan aturan kerja: [`AGENTS.md`](AGENTS.md).

## Status

Tahap fondasi, pipeline data, modul training, API FastAPI v1, dan demo kamera Streamlit selesai di sisi kode. Dataset mentah di `data/raw/` (35.168 gambar) sudah melalui audit, deduplikasi perceptual hash, dan split stratified 70/15/15 seed 42 menjadi 30.644 gambar. Hasil olahan ada di `data/processed/` (tidak di-commit), bersama `splits.csv` dan `dedup_removed.csv`; ringkasan audit ada di `results/data_audit.csv` dan `results/class_distribution.png`. Smoke test training 1 epoch pada 0,5% data berhasil dijalankan di CPU (1,31 menit; metrik smoke bukan hasil evaluasi). Baseline penuh belum dijalankan, sehingga `models/model.onnx` belum ada dan `/v1/predict` akan mengembalikan `503 MODEL_NOT_LOADED` sampai model diekspor. Semua nilai evaluasi tetap `null`. Test repository: **46 passed**.

## Persyaratan

- Python 3.10+
- GPU opsional (disarankan untuk training; Colab/Kaggle dapat dipakai)

## Menjalankan

Panduan lengkap ada di [`docs/PANDUAN_MENJALANKAN.md`](docs/PANDUAN_MENJALANKAN.md). Jalankan semua perintah dari root repository dengan virtual environment aktif.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ruff check .
black --check .
pytest -q
python -m src.data.prepare_data --config configs/data.yaml --root .
```

Smoke test training tanpa menulis `results/experiments.csv`:

```bash
python -m src.train.train --config configs/train.yaml --root . --run-id E0_smoke_ulang_yolov8n-cls_224 --epochs 1 --fraction 0.005 --batch 8 --no-record
```

Baseline E1 memakai konfigurasi `configs/train.yaml` dan memerlukan konfirmasi karena estimasinya 9–10 jam di CPU:

```bash
python -m src.train.train --config configs/train.yaml --root .
```

Menjalankan API dan demo kamera:

```bash
uvicorn src.api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

API tetap berjalan tanpa `models/model.onnx`, tetapi `/v1/health` akan melaporkan `model_loaded: false` dan `/v1/predict` mengembalikan `503 MODEL_NOT_LOADED`. Demo kamera membutuhkan API aktif dan model ONNX yang sudah diekspor untuk menampilkan prediksi nyata.

## Konfigurasi

| File | Isi |
| --- | --- |
| `configs/data.yaml` | seed, lokasi dataset, 9 kelas, rasio split, praproses, deduplikasi |
| `configs/train.yaml` | run ID, model, hyperparameter, output training |
| `configs/api.yaml` | alamat server, lokasi model, batas upload, `top_k`, pengaturan kamera demo |

Semua parameter dibaca dari berkas tersebut, bukan di-hard-code di kode.

## Struktur

```text
configs/   konfigurasi YAML
data/      dataset mentah, hasil olahan, foto dunia nyata (tidak di-commit)
notebooks/ eksplorasi
src/data/  deduplikasi, praproses, split
src/train/ training YOLO-cls
src/eval/  metrik, confusion matrix, ekspor
src/api/   FastAPI v1
src/utils/ seed, logging, I/O
app/       demo Streamlit
models/    best.pt, model.onnx, model_card.json (tidak di-commit)
results/   experiments.csv, CHANGELOG.md, grafik, log
tests/     pytest
docs/      rancangan, kontrak API, laporan, slide
```

## Kontrak ringkas

Kelas dengan urutan indeks tetap: `battery`, `cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`, `textile`, `trash`. API memakai prefix `/v1` dengan endpoint `/health`, `/classes`, `/model`, `/predict`, `/predict/batch`. Detail lengkap ada di `AGENTS.md` bagian 8.

## Sitasi Dataset

Israt Jahan, Sanjida Tasneem, Fahmida Rifha. *Merged Waste Classification Dataset (MWCD)*. Mendeley Data, V1. DOI: 10.17632/x863v66fv3.1. Lisensi CC BY 4.0.
