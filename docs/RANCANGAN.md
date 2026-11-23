# Rancangan Project Klasifikasi Sampah dengan YOLO-cls (MWCD)

Dokumen rencana kerja dan kontrak teknis project. Disusun 2026-11-23.

## 1. Sumber dan Prioritas

| Sumber | Peran | Perlakuan |
| --- | --- | --- |
| `Rancangan Project Klasifikasi Sampah dengan YOLO (MWCD).pdf` (10 halaman) | Rancangan awal tim: peran, jadwal, eksperimen, risiko, checklist | Menjadi dasar narasi dan cakupan kerja |
| `AGENTS.md` | Sumber kebenaran kontrak: API v1, artefak, struktur folder, DoD, keamanan | Mengikat; jika bertentangan dengan PDF, yang dipakai adalah `AGENTS.md` |

Keputusan yang sudah disepakati saat penyusunan dokumen ini:

1. Struktur kode mengikuti `AGENTS.md` (bertingkat di bawah `src/`), bukan struktur flat di PDF.
2. Lokasi foto dunia nyata adalah `data/realworld/`; folder `real_world/` yang sudah ada tidak dipakai dan tidak dihapus tanpa konfirmasi.
3. Dokumen ini disimpan di `docs/RANCANGAN.md`.
4. Cakupan sesi ini adalah dokumen saja; tidak ada kode, dependency, atau pelatihan yang diubah.

## 2. Ringkasan Project

Sistem klasifikasi jenis sampah dari satu gambar menggunakan YOLO mode classify (YOLOv8-cls / YOLO11-cls) yang dilatih sendiri dengan transfer learning. Praproses memakai OpenCV. Tidak memakai layanan AI pihak ketiga, model generatif, atau training dari nol.

Dataset: Merged Waste Classification Dataset (MWCD), 35.168 gambar RGB, 9 kelas, format satu gambar = satu label dengan folder per kelas. Karena tidak ada bounding box, project ini adalah klasifikasi, bukan deteksi objek.

Deliverable akhir:

- Repo yang bisa dijalankan ulang dari nol sesuai README.
- `models/best.pt`, `models/model.onnx`, `models/model_card.json`.
- Laporan evaluasi: metrik, confusion matrix, analisis error, uji dunia nyata.
- API inferensi FastAPI sesuai kontrak `AGENTS.md` bagian 8, aplikasi demo, video demo 1–2 menit.
- Laporan akhir dan slide presentasi.

Non-goals: deteksi banyak objek dalam satu gambar, training tanpa pretrained weights, API AI pihak ketiga, deployment produksi skala penuh.

## 3. Tujuan dan Kriteria Keberhasilan

| Metrik | Target minimum | Target baik |
| --- | --- | --- |
| Accuracy (test set) | ≥ 85% | ≥ 92% |
| Macro-F1 | ≥ 0,83 | ≥ 0,90 |
| F1 kelas terendah | ≥ 0,70 | ≥ 0,80 |
| Akurasi foto dunia nyata | ≥ 65% | ≥ 80% |
| Waktu inferensi (CPU, ONNX) | < 200 ms | < 100 ms |

Target bersifat perkiraan awal. Penyesuaian hanya boleh dilakukan setelah baseline selesai dan harus dicatat di `results/CHANGELOG.md`. Test set dikunci dan hanya dipakai sekali untuk evaluasi akhir model terpilih. Pemilihan model final berdasarkan macro-F1 di validation set dan uji dunia nyata, bukan test set.

## 4. Dataset

| Item | Nilai |
| --- | --- |
| Nama | Merged Waste Classification Dataset (MWCD) |
| Sumber | Mendeley Data, DOI 10.17632/x863v66fv3.1 |
| Lisensi | CC BY 4.0 (wajib mencantumkan sitasi) |
| Jumlah | 35.168 gambar RGB |
| Kelas (urutan tetap, abjad) | `battery`, `cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`, `textile`, `trash` |

Distribusi aktual hasil audit awal di `data/raw/`:

| Kelas | Jumlah |
| --- | ---: |
| battery | 1.727 |
| cardboard | 3.166 |
| glass | 4.678 |
| metal | 2.911 |
| organic | 2.547 |
| paper | 3.496 |
| plastic | 3.897 |
| textile | 10.964 |
| trash | 1.782 |
| **Total** | **35.168** |

Ketimpangan kelas nyata (terutama `textile` vs `battery`/`trash`) menjadi dasar eksperimen E6. Data mentah bersifat read-only.

## 5. Peran dan Jadwal

| Kode | Peran | Fokus utama |
| --- | --- | --- |
| A | Project Lead & Integrator | Koordinasi, repo, jadwal, integrasi, laporan akhir |
| B | Data Engineer | Audit data, deduplikasi, praproses OpenCV, split dataset |
| C | ML Engineer | Training YOLO-cls, eksperimen, tuning, penanganan imbalance |
| D | Evaluator & Deployment | Evaluasi, analisis error, ekspor model, API, aplikasi demo |

Jika tim hanya 3 orang, gabungkan C dan D menjadi satu peran ML & Evaluasi, atau D ke A. Rekomendasi PDF: Lead tetap fokus koordinasi, evaluasi dibagi B dan C, demo dikerjakan minggu 5–6.

| Minggu | A (Lead) | B (Data) | C (ML) | D (Eval & Deploy) |
| --- | --- | --- | --- | --- |
| 1 | Repo, aturan Git, jadwal, studi pustaka | Download & audit distribusi, cek gambar rusak | Setup lingkungan, pelajari Ultralytics, skeleton training | Studi metrik, template evaluasi, kerangka demo |
| 2 | Review audit, putuskan strategi imbalance | Dedup, praproses OpenCV, split, serah terima | Uji training 5 epoch untuk validasi pipeline | Script confusion matrix dan laporan per kelas |
| 3 | Bab latar belakang & metodologi | Dokumentasi praproses, cek data leakage | Baseline `yolov8n-cls`, 50 epoch | Evaluasi baseline di val, laporan awal |
| 4 | Bab data & eksperimen | Siapkan foto dunia nyata untuk uji domain gap | Eksperimen ukuran model, augmentasi, class weighting | Evaluasi tiap eksperimen, tabel perbandingan |
| 5 | Putuskan model final bersama tim, bab hasil | Bantu analisis error kelas bermasalah | Training ulang model final, simpan seed & konfigurasi | Evaluasi final di test, analisis error, ekspor ONNX |
| 6 | Integrasi, finalisasi laporan & slide | Finalisasi dokumentasi data | Finalisasi dokumentasi training, rapikan notebook | Finalisasi demo, rekam video demo |

## 6. Alur Kerja dan Serah Terima

```text
[B] Download & audit data
  -> [B] Deduplikasi (perceptual hash) + cek gambar rusak
  -> [B] Praproses OpenCV + stratified split (train/val/test)
  -> HANDOFF B->C (akhir minggu 2): folder dataset final + file split
  -> [C] Training baseline YOLO-cls (nano)
  -> [C] Eksperimen E1-E7 (satu variabel per run)
  -> HANDOFF C->D (akhir minggu 5): best.pt + log training + experiments.csv
  -> [D] Evaluasi test + confusion matrix + analisis error
  -> [D] Ekspor ONNX + benchmark + API + demo
  -> [A] Integrasi, uji foto dunia nyata, laporan & presentasi
```

C tidak boleh mulai training sebelum B menyerahkan dataset final. Selama menunggu, C menyiapkan script dan lingkungan, D menyiapkan kerangka demo dan template evaluasi, A menyiapkan repo dan kerangka laporan. Jika ada keterlambatan, Lead memutuskan pengurangan ruang lingkup.

## 7. Struktur Repository Target

```text
waste-yolo/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/                    # train.yaml, data.yaml, api.yaml
├── data/                       # tidak di-commit kecuali .gitkeep
│   ├── .gitkeep
│   ├── raw/
│   ├── processed/
│   │   ├── dataset/
│   │   │   ├── train/<kelas>/*.jpg
│   │   │   ├── val/<kelas>/*.jpg
│   │   │   └── test/<kelas>/*.jpg
│   │   └── splits.csv
│   └── realworld/
├── notebooks/
│   ├── 01_audit_data.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluasi.ipynb
├── src/
│   ├── data/                   # prepare_data.py, dedup.py, split.py
│   ├── train/                  # train.py
│   ├── eval/                   # evaluate.py
│   ├── api/                    # main.py, schemas.py
│   └── utils/                  # seed.py, logging.py, io.py
├── app/                        # streamlit_app.py
├── models/                     # best.pt, model.onnx, model_card.json
├── results/                    # experiments.csv, CHANGELOG.md, grafik, log
├── tests/                      # pytest, termasuk test_api_contract.py
└── docs/                       # RANCANGAN.md, api_contract.md, laporan, slide
```

Kepemilikan folder mengikuti `AGENTS.md` bagian 4: `src/data/` + `data/` (B), `src/train/` + `configs/` (C), `src/eval/` + `src/api/` + `app/` (D), `docs/` + `AGENTS.md` (A), `models/` (C & D).

## 8. Kontrak Artefak Antar-Modul

| Dari -> Ke | Artefak | Format wajib |
| --- | --- | --- |
| Data -> Training | `data/processed/dataset/` | 9 folder kelas persis, struktur `train/val/test` |
| Data -> Semua | `data/processed/splits.csv` | `filepath,class,split,sha1` |
| Training -> Evaluasi | `models/best.pt` | Bobot YOLO-cls dari run terpilih |
| Training -> Evaluasi | `results/experiments.csv` | `run_id,model,imgsz,epochs,batch,aug,imbalance,val_acc,val_macro_f1,train_time_min,notes` |
| Evaluasi -> API | `models/model.onnx` | Input `1x3x224x224`, output 9 logit/probabilitas |
| Semua -> Dokumentasi | `models/model_card.json` | Field di bawah |

```json
{
  "name": "waste-yolo-cls",
  "version": "1.0.0",
  "architecture": "yolov8n-cls",
  "input_size": 224,
  "classes": ["battery", "cardboard", "glass", "metal", "organic", "paper", "plastic", "textile", "trash"],
  "dataset": "MWCD v1 (DOI 10.17632/x863v66fv3.1)",
  "seed": 42,
  "train_run_id": "E5_...",
  "commit": "<git hash>",
  "metrics": {"val_macro_f1": null, "test_accuracy": null, "test_macro_f1": null},
  "known_limitations": []
}
```

Nilai `metrics` tetap `null` sampai evaluasi nyata dijalankan. Model yang sudah dirilis tidak ditimpa; buat versi baru.

## 9. Tahap Implementasi

### 9.1 Data (B)

1. Audit: jumlah per kelas, plot distribusi, resolusi, file korup, contoh gambar per kelas.
2. Deduplikasi: perceptual hash (`ImageHash`) sebelum split, terutama lintas sumber gabungan, untuk mencegah data leakage.
3. Praproses OpenCV: resize `224x224` dengan padding menjaga rasio, konversi BGR->RGB, opsional CLAHE. Tidak menormalisasi dua kali karena Ultralytics sudah melakukannya.
4. Split stratified 70% train / 15% val / 15% test, seed 42, disimpan ke `data/processed/splits.csv`.
5. Dataset uji dunia nyata: 100–200 foto kondisi nyata di `data/realworld/`, tidak pernah masuk training, tanpa wajah atau data pribadi yang dapat dikenali.

### 9.2 Training (C)

Perintah baseline:

```bash
yolo classify train data=data/processed/dataset model=yolov8n-cls.pt \
  epochs=50 imgsz=224 batch=32 patience=10 seed=42
```

Semua parameter dibaca dari `configs/*.yaml`, tidak di-hard-code. Penanganan imbalance: oversampling kelas minoritas atau class weighting (E6). Setiap run wajib menyimpan YAML, log training, kurva loss/accuracy, waktu training, versi library, dan commit hash. Pola nama run: `E{nomor}_{model}_{imgsz}_{catatan}`, contoh `E2_yolov8s-cls_224_baseline-aug`.

### 9.3 Evaluasi (D)

Hitung accuracy, precision, recall, F1 per kelas (macro dan weighted), confusion matrix, dan waktu inferensi. Analisis error: kumpulkan gambar salah prediksi, kelompokkan penyebabnya (kelas mirip, gambar buram, label meragukan), berikan rekomendasi. Uji dunia nyata dibandingkan dengan akurasi test set untuk mengukur domain gap. Hasil ditulis ke `results/experiments.csv`, `results/CHANGELOG.md`, dan laporan.

### 9.4 Ekspor ONNX dan Benchmark

Ekspor model final ke `models/model.onnx` dengan input `1x3x224x224` dan 9 output. Ukur latency CPU per gambar (ms) dan bandingkan dengan target `< 200 ms` / `< 100 ms`. Hasil benchmark dicatat, bukan diperkirakan.

### 9.5 API Inferensi v1 (D)

| Method | Path | Fungsi |
| --- | --- | --- |
| GET | `/v1/health` | Cek layanan dan model termuat |
| GET | `/v1/classes` | Daftar 9 kelas dan indeksnya |
| GET | `/v1/model` | Metadata model aktif |
| POST | `/v1/predict` | Prediksi satu gambar |
| POST | `/v1/predict/batch` | Prediksi beberapa gambar (maks. 16) |

Perilaku wajib: model dimuat sekali saat startup; `top_k` 1–9 (default 3); confidence float 0–1, total probabilitas = 1 ± 0,001; `top_k` terurut menurun; praproses identik dengan training; field `snake_case`; log request tanpa menyimpan gambar pengguna.

Format error seragam:

```json
{"error": {"code": "INVALID_IMAGE", "message": "File bukan gambar yang valid.", "details": null}}
```

| HTTP | Code | Kondisi |
| --- | --- | --- |
| 400 | `INVALID_IMAGE` | File rusak atau bukan gambar |
| 400 | `INVALID_PARAMETER` | `top_k` di luar 1–9, dll. |
| 413 | `FILE_TOO_LARGE` | Ukuran > 5 MB |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Format selain JPG/PNG |
| 422 | `VALIDATION_ERROR` | Field wajib tidak ada |
| 429 | `RATE_LIMITED` | Terlalu banyak permintaan (jika diaktifkan) |
| 500 | `INTERNAL_ERROR` | Kesalahan tak terduga (tanpa stack trace) |
| 503 | `MODEL_NOT_LOADED` | Model belum siap |

Skema Pydantic acuan ada di `AGENTS.md` bagian 8.6. Perubahan kontrak wajib memperbarui `AGENTS.md`, `docs/api_contract.md`, `src/api/schemas.py`, dan `tests/test_api_contract.py`; perubahan merusak mewajibkan prefix `/v2`.

### 9.6 Aplikasi Demo (D)

Streamlit memanggil API, tidak memuat model sendiri. Fitur: upload gambar, tampilan kelas prediksi dengan top-3 confidence, opsional webcam via OpenCV. Rekam video demo 1–2 menit.

## 10. Rencana Eksperimen

| ID | Variabel yang diubah | Tujuan |
| --- | --- | --- |
| E1 | Baseline: `yolov8n-cls`, augmentasi default | Titik acuan |
| E2 | Ukuran model: `yolov8s-cls` | Pengaruh kapasitas model |
| E3 | Ukuran model: `yolov8m-cls` | Batas akurasi vs biaya komputasi |
| E4 | Resolusi 224 vs 320 | Pengaruh detail gambar |
| E5 | Augmentasi lebih kuat | Mengurangi overfitting dan domain gap |
| E6 | Penanganan imbalance (oversampling/weighting) | Meningkatkan recall kelas minoritas |
| E7 | YOLOv8-cls vs YOLO11-cls | Memilih arsitektur terbaik |

Satu variabel per eksperimen. Pemilihan final berdasarkan macro-F1 validation dan uji dunia nyata.

## 11. Testing dan Definition of Done

Perintah standar:

```bash
pip install -r requirements.txt
ruff check . && black --check .
pytest -q
uvicorn src.api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

Sebuah tugas selesai jika: kode berjalan dan lolos ruff/black; test relevan ada dan lolos; tidak ada secret, data besar, atau model ter-commit; kontrak terpenuhi atau perubahannya disetujui dan didokumentasikan; angka berasal dari eksekusi nyata; README/dokumen diperbarui; sudah direview lewat PR.

Test kontrak API minimal: `/health` 200; `/classes` 9 kelas urutan benar; `/predict` gambar valid sesuai struktur 8.3; file bukan gambar 400 `INVALID_IMAGE`; file > 5 MB 413 `FILE_TOO_LARGE`.

## 12. Reproduksibilitas, Keamanan, dan Privasi

- Seed tetap 42 untuk Python, NumPy, PyTorch, dan split data.
- Versi library dikunci di `requirements.txt`; setiap hasil ditelusurkan ke konfigurasi dan commit hash.
- Tidak meng-commit dataset, model besar, `.env`, API key, atau kredensial.
- Validasi API: tipe file, ukuran, dapat dibuka sebagai gambar. Error tidak membocorkan path server atau stack trace.
- Gambar unggahan tidak disimpan secara default; foto dunia nyata bebas wajah dan data pribadi.
- Aturan Git: `main` selalu bisa dijalankan; branch `feat/...`, `fix/...`, `docs/...`; PR direview minimal satu orang; Conventional Commits; tanpa push ke `main` atau force-push oleh agen.

## 13. Risiko dan Antisipasi

| Risiko | Dampak | Antisipasi | Penanggung jawab |
| --- | --- | --- | --- |
| Kelas tidak seimbang | Recall kelas minoritas rendah | Oversampling/weighting, laporkan F1 per kelas | C |
| Data leakage akibat dataset gabungan | Akurasi tinggi tapi menyesatkan | Dedup hash sebelum split | B |
| Domain gap (latar bersih vs dunia nyata) | Akurasi turun di lapangan | Foto sendiri, augmentasi kuat | B, D |
| Kelas ambigu (`trash`, `plastic`, `paper` vs `cardboard`) | Banyak salah prediksi | Analisis error, diskusi di laporan | D |
| Keterbatasan GPU | Training lambat | Model nano/small, Colab/Kaggle, kurangi epoch | C |
| Anggota terlambat | Jadwal mundur | Handoff jelas, buffer minggu 6 | A |
| Hasil tidak reproducible | Laporan diragukan | Seed tetap, versi locked, log eksperimen | C, A |

## 14. Checklist Deliverable Akhir

- [ ] Repo GitHub dengan README dan instruksi menjalankan ulang
- [ ] Notebook audit data dan script praproses
- [ ] Dataset final dengan daftar split tanpa duplikat lintas split
- [ ] Notebook/script training dan tabel eksperimen E1–E7
- [ ] Model final (`best.pt`, ONNX, `model_card.json`)
- [ ] Laporan evaluasi: metrik, confusion matrix, analisis error, uji dunia nyata
- [ ] Aplikasi demo dan video demo
- [ ] Laporan akhir dan slide presentasi
- [ ] Sitasi dataset sesuai CC BY 4.0

## 15. Pengembangan Lanjutan (Opsional)

Jika waktu tersisa: object detection dengan anotasi bounding box pada subset gambar (Roboflow/LabelImg) agar beberapa sampah dalam satu foto dikenali sekaligus; visualisasi Grad-CAM untuk menjelaskan keputusan model. Keduanya di luar cakupan saat ini.

## 16. Kondisi Awal Repository (Audit Saat Penyusunan)

| Item | Kondisi aktual |
| --- | --- |
| `AGENTS.md`, PDF rancangan | Ada, belum ter-commit |
| `README.md`, `src/*.py`, `notebooks/*.ipynb`, `app/streamlit_app.py` | Ada tetapi kosong (0 byte) |
| `requirements.txt` | 13 dependency terpasang: ultralytics, opencv-python-headless, numpy, pandas, scikit-learn, matplotlib, seaborn, ImageHash, Pillow, onnx, onnxruntime, streamlit, tqdm |
| Dependency yang dirujuk `AGENTS.md` tetapi belum ada | fastapi, uvicorn, python-multipart, pytest, ruff, black, PyYAML |
| `configs/`, `tests/`, `docs/api_contract.md`, `docs/RANCANGAN.md` | Belum ada sebelum dokumen ini dibuat |
| `data/raw/` | Sudah berisi 35.168 gambar sesuai distribusi di bagian 4 |
| `data/processed/` | Belum ada |
| `data/dataset/`, `data/train/`, `data/val/`, `data/test/`, `real_world/` | Kosong; tidak dihapus tanpa konfirmasi (di luar struktur target) |
| `models/`, `results/` | Hanya `.gitkeep` |
| `.gitignore` | Tidak aktif; yang ada hanya `.gitignore~` (backup, ter-commit) sehingga aturan ignore belum berlaku |

Gap di atas menjadi pekerjaan implementasi berikutnya dan **tidak** diubah oleh dokumen ini. Penambahan dependency dan pengaktifan `.gitignore` memerlukan konfirmasi Lead sebelum dikerjakan.

## 17. Keputusan Manusia yang Masih Diperlukan

1. Menyetujui penambahan dependency yang belum ada di `requirements.txt` (fastapi, uvicorn, python-multipart, pytest, ruff, black, PyYAML).
2. Menyetujui pembuatan `.gitignore` aktif dan perlakuan terhadap folder kosong legacy.
3. Menyetujui training runs yang diperkirakan > 30 menit atau berbiaya.
4. Memilih model final berdasarkan hasil evaluasi nyata.
5. Menyetujui penyesuaian target metrik jika dilakukan setelah baseline (dicatat di `results/CHANGELOG.md`).

## 18. Sitasi Dataset

Israt Jahan, Sanjida Tasneem, Fahmida Rifha. *Merged Waste Classification Dataset (MWCD)*. Mendeley Data, V1. DOI: 10.17632/x863v66fv3.1. Lisensi CC BY 4.0.
