AGENTS.md – Panduan & Kontrak untuk OpenCode
Project Klasifikasi Sampah dengan YOLO-cls (dataset MWCD) · Dokumen ini ditujukan untuk agen AI coding (OpenCode) dan seluruh anggota tim
Cara memakai: salin isi dokumen ini ke file AGENTS.md di root repository agar OpenCode membacanya otomatis di setiap sesi. Jika ada aturan yang berubah, ubah di sini dan sinkronkan ke AGENTS.md. Dokumen ini adalah sumber kebenaran: jika instruksi lisan bertentangan dengan dokumen ini, agen harus bertanya dulu, bukan menebak.
1. Penjelasan Project
1.1 Apa project ini
Sistem klasifikasi jenis sampah dari gambar menggunakan YOLO mode classify (YOLOv8-cls / YOLO11-cls) yang dilatih sendiri (transfer learning) pada dataset Merged Waste Classification Dataset (MWCD). Praproses gambar memakai computer vision klasik (OpenCV). Tidak memakai layanan AI siap pakai atau model generatif untuk proses klasifikasinya.
1.2 Dataset
Item
Nilai
Nama
Merged Waste Classification Dataset (MWCD)
Sumber
Mendeley Data, DOI 10.17632/x863v66fv3.1
Lisensi
CC BY 4.0 (wajib mencantumkan sitasi)
Jumlah gambar
35.168 gambar RGB
Format
Image classification (satu gambar = satu label, folder per kelas)
Kelas (9)
battery, cardboard, glass, metal, organic, paper, plastic, textile, trash
Dataset tidak memiliki bounding box, sehingga project ini adalah klasifikasi, bukan object detection.
1.3 Tujuan
1.  Membangun model klasifikasi 9 kelas sampah dengan akurasi yang dapat dipertanggungjawabkan.
2.  Menyusun pipeline yang dapat direproduksi (data → training → evaluasi → deploy).
3.  Menyediakan layanan inferensi (API) dan aplikasi demo agar model dapat dicoba langsung.
4.  Menghasilkan dokumentasi dan laporan evaluasi yang jujur, termasuk kelemahan model.
1.4 Di luar cakupan (non-goals)
• Deteksi banyak objek sekaligus dalam satu gambar (hanya opsional di pengembangan lanjutan)
• Pelatihan dari nol tanpa pretrained weights
• Penggunaan API AI pihak ketiga untuk prediksi
• Deployment skala produksi (autoscaling, monitoring penuh)
1.5 Hasil capaian yang diharapkan
Metrik
Target minimum
Target baik
Accuracy (test set)
≥ 85%
≥ 92%
Macro-F1
≥ 0,83
≥ 0,90
F1 kelas terendah
≥ 0,70
≥ 0,80
Akurasi pada foto dunia nyata
≥ 65%
≥ 80%
Waktu inferensi (CPU, ONNX)
< 200 ms
< 100 ms
Target adalah perkiraan awal; disesuaikan setelah baseline selesai dan dicatat perubahannya di results/CHANGELOG.md.
Deliverable akhir:
• Repo yang bisa dijalankan ulang dari nol sesuai README
• models/best.pt dan models/model.onnx beserta models/model_card.json
• Laporan evaluasi (metrik, confusion matrix, analisis error, uji dunia nyata)
• API inferensi (FastAPI) sesuai kontrak di bagian 8
• Aplikasi demo (Streamlit/Gradio) dan video demo 1–2 menit
• Laporan akhir dan slide presentasi
2. Peran Agen (OpenCode)
Agen bekerja sebagai asisten engineer di repo ini, bukan pengambil keputusan arsitektur. Agen boleh menulis kode, menjalankan perintah yang aman, menulis test, dan menyusun dokumentasi. Keputusan yang mengubah desain, kontrak, target metrik, atau struktur repo harus dikonfirmasi manusia (Project Lead) terlebih dahulu.
Prinsip kerja agen:
1.  Baca dulu, ubah kemudian. Lihat file terkait dan struktur folder sebelum mengedit.
2.  Perubahan kecil dan terfokus. Satu tugas = satu perubahan yang bisa direview.
3.  Verifikasi sebelum menyatakan selesai. Jalankan test/lint/skrip terkait dan laporkan hasil sebenarnya, termasuk jika gagal.
4.  Jangan mengarang. Jangan menulis angka metrik, hasil eksperimen, atau nama file yang tidak benar-benar ada atau dijalankan.
5.  Bertanya saat ragu. Lebih baik satu pertanyaan singkat daripada asumsi yang salah.
3. Tech Stack
Kebutuhan
Pilihan
Catatan
Bahasa
Python 3.10+
Kunci versi di requirements.txt
Model
Ultralytics YOLO (mode classify)
Mulai dari yolov8n-cls.pt
Praproses
OpenCV, NumPy, Pillow

Analisis
Pandas, Matplotlib, scikit-learn

API
FastAPI + Uvicorn
Sesuai kontrak bagian 8
Demo
Streamlit atau Gradio
Memanggil API, bukan memuat model sendiri
Ekspor/inferensi
ONNX Runtime

Test
pytest

Lint/format
ruff + black

Penambahan dependency baru harus dicatat di requirements.txt dan disebutkan alasannya di deskripsi pull request.
4. Struktur Repository dan Kepemilikan Folder
waste-yolo/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/              # konfigurasi YAML (train, data, api)
├── data/                 # TIDAK di-commit (hanya .gitkeep + instruksi)
├── notebooks/            # eksplorasi; bukan sumber logika utama
├── src/
│   ├── data/             # prepare_data.py, dedup.py, split.py
│   ├── train/            # train.py
│   ├── eval/             # evaluate.py
│   ├── api/              # main.py, schemas.py
│   └── utils/            # seed.py, logging.py, io.py
├── app/                  # streamlit_app.py
├── models/               # best.pt, model.onnx, model_card.json (LFS/Drive)
├── results/              # tabel, grafik, log eksperimen
├── tests/                # pytest
└── docs/                 # laporan, slide, kontrak API
Folder
Pemilik
Aturan untuk agen
src/data/, data/
Data Engineer (B)
Ubah hanya saat tugasnya data; jangan hapus data mentah
src/train/, configs/
ML Engineer (C)
Satu variabel per eksperimen
src/eval/, src/api/, app/
Evaluator & Deployment (D)
Patuhi kontrak API
docs/, AGENTS.md
Lead (A)
Jangan ubah tanpa diminta
models/
C & D
Jangan menimpa model yang sudah dirilis; buat versi baru
5. Aturan Umum untuk Agen
5.1 Wajib (DO)
• Gunakan seed tetap 42 untuk semua proses acak (Python, NumPy, PyTorch, split data).
• Baca parameter dari configs/*.yaml, jangan menulis nilai di dalam kode (hard-code).
• Gunakan type hints dan docstring singkat pada fungsi publik.
• Gunakan pathlib.Path, bukan string path manual.
• Gunakan logging, bukan print, pada kode di src/.
• Tulis test untuk logika baru (minimal happy path + satu kasus gagal).
• Perbarui README atau dokumen terkait jika perilaku atau perintah berubah.
• Setelah selesai, laporkan sesuai format di bagian 12.
5.2 Dilarang (DON'T)
• Jangan meng-commit dataset, file model besar, .env, API key, atau kredensial.
• Jangan menjalankan perintah destruktif (rm -rf, git reset --hard, git push --force, menimpa folder data/) tanpa konfirmasi eksplisit manusia.
• Jangan menyentuh test set untuk keputusan tuning atau pemilihan model.
• Jangan mengubah target metrik, kontrak API, atau urutan kelas secara diam-diam.
• Jangan menambah dependency besar atau mengganti framework tanpa persetujuan.
• Jangan mengisi tabel hasil dengan angka perkiraan; hanya angka yang benar-benar dihitung.
• Jangan menyalin kode berlisensi tidak jelas ke repo.
5.3 Perlu konfirmasi manusia sebelum dikerjakan
• Mengubah struktur folder atau nama file yang dirujuk dokumen lain
• Mengubah skema request/response API (lihat aturan versioning bagian 8.7)
• Menghapus atau memindahkan data atau model
• Menjalankan training panjang (> 30 menit) atau yang berbiaya
• Memilih model final
6. Aturan Data
1.  Data mentah bersifat read-only. Simpan di data/raw/; hasil olahan di data/processed/.
2.  Deduplikasi sebelum split. Hitung hash gambar (perceptual hash) dan buang duplikat, terutama lintas sumber dataset, untuk mencegah data leakage.
3.  Split stratified 70% train / 15% val / 15% test, seed=42. Simpan daftar file tiap split di data/processed/splits.csv.
4.  Test set dikunci. Hanya dipakai satu kali untuk evaluasi akhir model terpilih.
5.  Praproses konsisten antara training dan inferensi: resize 224×224 (padding menjaga rasio), warna RGB. Jangan menormalisasi dua kali karena Ultralytics sudah melakukannya.
6.  Format dataset untuk YOLO-cls:
data/processed/dataset/
├── train/<nama_kelas>/*.jpg
├── val/<nama_kelas>/*.jpg
└── test/<nama_kelas>/*.jpg
7.  Dataset uji dunia nyata (foto sendiri, 100–200 gambar) disimpan terpisah di data/realworld/ dan tidak pernah masuk ke train.
8.  Cantumkan sitasi dataset di README dan laporan sesuai CC BY 4.0.
7. Aturan Training dan Eksperimen
1.  Satu variabel per eksperimen agar hasil dapat dibandingkan.
2.  Setiap run diberi nama dengan pola E{nomor}_{model}_{imgsz}_{catatan}, contoh E2_yolov8s-cls_224_baseline-aug.
3.  Setiap run wajib menyimpan: konfigurasi YAML, log training, kurva loss/accuracy, waktu training, versi library, dan commit hash.
4.  Perintah baseline:
yolo classify train data=data/processed/dataset model=yolov8n-cls.pt epochs=50 imgsz=224 batch=32 patience=10 seed=42
5.  Pemilihan model final berdasarkan macro-F1 di validation set dan hasil uji dunia nyata, bukan test set.
6.  Hasil eksperimen dicatat di results/experiments.csv dengan kolom: run_id, model, imgsz, epochs, batch, aug, imbalance, val_acc, val_macro_f1, train_time_min, notes.
8. Kontrak API (Inferensi)
Layanan inferensi dibangun dengan FastAPI. Aplikasi demo dan klien lain hanya berinteraksi lewat kontrak ini. Perubahan harus mengikuti aturan versioning di 8.7.
8.1 Informasi dasar
Item
Nilai
Base URL (lokal)
http://localhost:8000
Prefix versi
/v1
Format
JSON (application/json), kecuali upload gambar (multipart/form-data)
Encoding
UTF-8
Autentikasi
Tidak ada (lokal/demo). Jika dibuka publik, tambahkan API key lewat header X-API-Key
Dokumentasi otomatis
/docs (Swagger) dan /openapi.json
8.2 Daftar endpoint
Method
Path
Fungsi
GET
/v1/health
Cek layanan dan model termuat
GET
/v1/classes
Daftar kelas dan indeksnya
GET
/v1/model
Metadata model aktif
POST
/v1/predict
Prediksi satu gambar
POST
/v1/predict/batch
Prediksi beberapa gambar (maks. 16)
8.3 Detail endpoint
GET /v1/health → 200 OK
{
  "status": "ok",
  "model_loaded": true,
  "api_version": "v1"
}
GET /v1/classes → 200 OK
Urutan indeks tetap dan mengikuti urutan abjad nama folder (sama dengan YOLO-cls).
{
  "classes": [
    {"index": 0, "name": "battery"},
    {"index": 1, "name": "cardboard"},
    {"index": 2, "name": "glass"},
    {"index": 3, "name": "metal"},
    {"index": 4, "name": "organic"},
    {"index": 5, "name": "paper"},
    {"index": 6, "name": "plastic"},
    {"index": 7, "name": "textile"},
    {"index": 8, "name": "trash"}
  ]
}
GET /v1/model → 200 OK
{
  "name": "waste-yolo-cls",
  "version": "1.0.0",
  "architecture": "yolov8n-cls",
  "input_size": 224,
  "num_classes": 9,
  "format": "onnx",
  "trained_on": "MWCD v1",
  "metrics": {"test_accuracy": null, "test_macro_f1": null}
}
Nilai metrics diisi dari hasil evaluasi nyata; tetap null sebelum evaluasi dilakukan.
POST /v1/predict → 200 OK
Request: multipart/form-data dengan field file (gambar JPG/PNG, maks. 5 MB). Query opsional: top_k (integer 1–9, default 3).
Response:
{
  "predicted_class": "plastic",
  "confidence": 0.93,
  "top_k": [
    {"class": "plastic", "confidence": 0.93},
    {"class": "glass", "confidence": 0.04},
    {"class": "metal", "confidence": 0.02}
  ],
  "inference_ms": 41.7,
  "model_version": "1.0.0"
}
POST /v1/predict/batch → 200 OK
Request: multipart/form-data dengan field files (1–16 gambar). Response:
{
  "results": [
    {
      "filename": "a.jpg",
      "predicted_class": "paper",
      "confidence": 0.88,
      "top_k": [{"class": "paper", "confidence": 0.88}]
    }
  ],
  "inference_ms": 120.3,
  "model_version": "1.0.0"
}
Jika satu file gagal diproses, file itu mengembalikan objek error (format 8.4) pada posisinya, dan file lain tetap diproses.
8.4 Format error (seragam)
Semua error memakai bentuk berikut:
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "File bukan gambar yang valid.",
    "details": null
  }
}
HTTP
code
Kapan terjadi
400
INVALID_IMAGE
File rusak atau bukan gambar
400
INVALID_PARAMETER
top_k di luar 1–9, dll.
413
FILE_TOO_LARGE
Ukuran > 5 MB
415
UNSUPPORTED_MEDIA_TYPE
Format selain JPG/PNG
422
VALIDATION_ERROR
Field wajib tidak ada
429
RATE_LIMITED
Terlalu banyak permintaan (jika diaktifkan)
500
INTERNAL_ERROR
Kesalahan tak terduga (jangan bocorkan stack trace)
503
MODEL_NOT_LOADED
Model belum siap
8.5 Aturan perilaku API
• confidence bertipe float 0–1; jumlah seluruh probabilitas kelas = 1 (±0,001).
• top_k diurutkan menurun berdasarkan confidence.
• Praproses di API identik dengan praproses training (resize 224, RGB).
• Nama field memakai snake_case; nama kelas huruf kecil bahasa Inggris persis seperti di 8.3.
• Model dimuat sekali saat startup, bukan per request.
• Log setiap request (waktu, endpoint, status, durasi) tanpa menyimpan gambar pengguna.
8.6 Skema Pydantic (acuan di src/api/schemas.py)
from pydantic import BaseModel, Field

class ClassScore(BaseModel):
    class_: str = Field(alias="class")
    confidence: float = Field(ge=0, le=1)

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    top_k: list[ClassScore]
    inference_ms: float
    model_version: str
8.7 Versioning dan perubahan kontrak
• Perubahan yang tidak merusak (menambah field opsional) boleh dalam v1 setelah disetujui Lead.
• Perubahan yang merusak (menghapus/mengubah nama field, mengubah makna, mengubah urutan kelas) mewajibkan prefix baru /v2 dan dokumentasi migrasi.
• Setiap perubahan kontrak wajib memperbarui: dokumen ini, docs/api_contract.md, skema Pydantic, dan test kontrak di tests/test_api_contract.py.
9. Kontrak Antar-Modul (Artefak)
Modul berkomunikasi lewat file dengan format tetap berikut.
Dari → Ke
Artefak
Format wajib
Data → Training
data/processed/dataset/
Struktur folder bagian 6, nama kelas persis 9 kelas
Data → Semua
data/processed/splits.csv
Kolom: filepath, class, split, sha1
Training → Evaluasi
models/best.pt
Bobot YOLO-cls dari run terpilih
Training → Evaluasi
results/experiments.csv
Kolom bagian 7
Evaluasi → API
models/model.onnx
Input 1×3×224×224, output 9 logit/probabilitas
Semua → Dokumentasi
models/model_card.json
Lihat di bawah
Isi model_card.json:
{
  "name": "waste-yolo-cls",
  "version": "1.0.0",
  "architecture": "yolov8n-cls",
  "input_size": 224,
  "classes": ["battery","cardboard","glass","metal","organic","paper","plastic","textile","trash"],
  "dataset": "MWCD v1 (DOI 10.17632/x863v66fv3.1)",
  "seed": 42,
  "train_run_id": "E5_...",
  "commit": "<git hash>",
  "metrics": {"val_macro_f1": null, "test_accuracy": null, "test_macro_f1": null},
  "known_limitations": []
}
10. Git dan Kolaborasi
• Branch main selalu dalam keadaan bisa dijalankan. Kerjakan di branch: feat/data-..., feat/train-..., feat/eval-..., feat/api-..., fix/..., docs/....
• Commit message (Conventional Commits): feat(data): tambah deduplikasi hash, fix(api): tangani file korup, docs: perbarui kontrak API.
• Merge lewat pull request dan direview minimal satu anggota.
• Agen tidak melakukan push ke main dan tidak force-push.
• Deskripsi PR memuat: apa yang berubah, alasan, cara menguji, dan dampak ke kontrak (jika ada).
11. Testing dan Definition of Done
Perintah standar:
pip install -r requirements.txt
ruff check . && black --check .
pytest -q
uvicorn src.api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
Sebuah tugas dianggap selesai jika:
[ ] Kode berjalan tanpa error dan lolos ruff/black
[ ] Test relevan ada dan lolos (pytest)
[ ] Tidak ada secret, data besar, atau model yang ikut ter-commit
[ ] Kontrak (API/artefak) tetap terpenuhi atau perubahannya sudah disetujui dan didokumentasikan
[ ] Hasil/angka yang dilaporkan benar-benar berasal dari eksekusi
[ ] Dokumentasi/README diperbarui jika perlu
[ ] Sudah direview anggota tim melalui PR
Test kontrak API minimal: /health mengembalikan 200; /classes mengembalikan 9 kelas dengan urutan benar; /predict dengan gambar valid mengembalikan struktur 8.3; file bukan gambar mengembalikan 400 INVALID_IMAGE; file > 5 MB mengembalikan 413.
12. Format Laporan Agen Setelah Tugas
Setelah menyelesaikan tugas, agen melaporkan singkat dengan struktur berikut:
1.  Yang dikerjakan: daftar file yang dibuat/diubah
2.  Cara verifikasi: perintah yang dijalankan dan hasil sebenarnya (lolos/gagal)
3.  Asumsi: hal yang diasumsikan agen
4.  Risiko/keterbatasan: yang belum teruji atau masih kurang
5.  Perlu keputusan manusia: daftar pertanyaan (jika ada)
Jika sebuah test gagal atau tugas tidak bisa diselesaikan, agen menyatakannya secara jelas, bukan menyembunyikannya.
13. Keamanan dan Privasi
• Jangan menulis secret ke kode; gunakan variabel lingkungan dan .env (masuk .gitignore).
• Validasi semua input file di API (tipe, ukuran, dapat dibuka sebagai gambar).
• Gambar yang diunggah pengguna tidak disimpan secara default.
• Pesan error tidak membocorkan path server atau stack trace.
• Foto dunia nyata yang dikumpulkan tim tidak boleh memuat wajah atau data pribadi yang dapat dikenali.
14. Milestone dan Pembagian Tanggung Jawab
Minggu
Milestone
Penanggung jawab utama
1
Repo, audit data, setup lingkungan
A, B
2
Dataset final + split (serah terima ke training)
B
3
Baseline YOLO-cls selesai
C
4
Eksperimen E2–E7
C, D
5
Model final, evaluasi test, ekspor ONNX
C, D
6
API + demo + laporan + presentasi
D, A
Peran: A = Project Lead & Integrator, B = Data Engineer, C = ML Engineer, D = Evaluator & Deployment.
15. Glosarium Singkat
Istilah
Arti
YOLO-cls
Mode klasifikasi gambar pada Ultralytics YOLO
Transfer learning
Memakai bobot pretrained lalu dilatih lanjut pada dataset sendiri
Data leakage
Gambar yang sama/mirip muncul di train dan test sehingga akurasi menyesatkan
Macro-F1
Rata-rata F1 antar kelas dengan bobot sama, sensitif terhadap kelas minoritas
Domain gap
Perbedaan kondisi gambar dataset dengan kondisi dunia nyata
ONNX
Format model portabel untuk inferensi yang cepat
16. Sitasi Dataset
Israt Jahan, Sanjida Tasneem, Fahmida Rifha. Merged Waste Classification Dataset (MWCD). Mendeley Data, V1. DOI: 10.17632/x863v66fv3.1. Lisensi CC BY 4.0.