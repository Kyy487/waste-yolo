# Handoff Pekerjaan — Branch `riezky`

Dokumen ini merangkum apa yang sudah selesai dan apa yang harus dilanjutkan oleh tim. Branch aktif: `riezky`. Commit terakhir saat handoff dibuat: `7c212d5`.

## Status saat ini

| Bagian | Status | Keterangan |
| --- | --- | --- |
| Fondasi repository | Selesai | `.gitignore`, `requirements.txt`, struktur `src/`, notebook valid |
| Pipeline data | Selesai | 35.168 gambar → 30.644 gambar; split 70/15/15 seed 42 |
| Modul training | Selesai | `src/train/train.py` + test; smoke test CPU berhasil |
| Rancangan real-time | Selesai | `docs/RANCANGAN_REALTIME.md`, mode klasifikasi per frame |
| Kontrak API referensi | Selesai | `docs/api_contract.md`, bukan perubahan skema |
| Model final | Belum ada | Baseline E1 belum dijalankan penuh |
| Evaluasi test | Belum ada | Test set terkunci, hanya dipakai sekali |
| ONNX | Belum ada | `models/model.onnx` belum diekspor |
| API FastAPI | Selesai (kode + test) | `src/api/main.py`, `schemas.py`, `inference.py`; butuh `models/model.onnx` untuk prediksi nyata |
| Demo kamera | Selesai (kode + test) | `app/api_client.py`, `app/streamlit_app.py`; pola satu frame per rerun |

## Commit yang sudah dibuat

1. `152b55a` — `chore: siapkan fondasi repository`
2. `21e5af6` — `docs: tambah rancangan project dan kamera real-time`
3. `2aaddd5` — `feat(data): tambah pipeline audit, dedup, dan split MWCD`
4. `ee252c2` — `feat(train): tambah training YOLO-cls dengan artefak run`
5. `1aaed3c` — `chore: siapkan placeholder API dan evaluasi`
6. `7c212d5` — `docs: tambah ringkasan audit dan panduan menjalankan`
7. Paket handoff ini — `docs: siapkan kontrak API dan paket handoff`

## Yang harus dikerjakan berikutnya

### C — ML Engineer

- [ ] Konfirmasi dulu apakah baseline E1 dijalankan di GPU atau CPU. Di CPU estimasinya 9–10 jam, jadi butuh persetujuan.
- [ ] Jalankan `python -m src.train.train --config configs/train.yaml --root .`.
- [ ] Catat hasil nyata ke `results/experiments.csv`. Jangan mengisi angka perkiraan.
- [ ] Lakukan eksperimen E2–E7 dengan satu variabel per eksperimen.
- [ ] Pilih model final berdasarkan macro-F1 validation dan uji dunia nyata, bukan test set.
- [ ] Ekspor `models/model.onnx` dan ukur inference aktual.
- [ ] Isi `models/model_card.json`, termasuk commit hash dan metrik nyata.

### D — Evaluator & Deployment

- [ ] Evaluasi test set **sekali** untuk model terpilih: accuracy, macro-F1, confusion matrix, analisis error.
- [ ] Isi `results/CHANGELOG.md` dengan hasil dan keterbatasan nyata.
- [x] Implementasi API v1: `src/api/main.py`, `src/api/schemas.py`, `src/api/inference.py`.
- [x] Tulis `tests/test_api_contract.py` sesuai `AGENTS.md` bagian 11.
- [x] Tambahkan `httpx` ke `requirements.txt` untuk `TestClient`.
- [x] Implementasi demo kamera: `app/api_client.py`, `app/streamlit_app.py`, bagian `demo:` di `configs/api.yaml`.
- [x] Tambahkan `requests` ke `requirements.txt` untuk multipart dari Streamlit ke API.
- [ ] Rekam video demo 1–2 menit (menunggu model ONNX).

### B — Data Engineer

- [ ] Bantu kumpulkan 100–200 foto di `data/realworld/`, tanpa wajah atau data pribadi.
- [ ] Pastikan foto dunia nyata tidak masuk ke train/val/test.

### A — Project Lead

- [ ] Review PR `riezky → main` minimal oleh satu anggota.
- [ ] Setujui dependency tambahan `requests` dan `httpx`.
- [ ] Setujui bagian `demo:` di `configs/api.yaml`.
- [ ] Setujui model final setelah evaluasi selesai.

## Urutan ketergantungan

```text
rancangan disetujui
  → model final + ONNX
    → evaluasi test sekali
      → API v1 + test kontrak
        → demo kamera + video
          → PR review
```

## Perintah untuk melanjutkan

```bash
git checkout riezky
git pull origin riezky

pip install -r requirements.txt
ruff check .
black --check .
pytest -q

python -m src.train.train --config configs/train.yaml --root .
uvicorn src.api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

Perintah `uvicorn` dan `streamlit` sudah bisa dijalankan. Prediksi nyata tetap menunggu `models/model.onnx`; tanpa model, API mengembalikan `503 MODEL_NOT_LOADED`.

## Catatan penting

- Jangan commit `data/raw/`, `data/processed/dataset/`, `models/*.pt`, `models/*.onnx`, `.venv/`, atau `.env`.
- Jangan commit `.gitignore~`; file itu sudah ada sebelum sesi ini.
- Test set hanya untuk evaluasi akhir, bukan tuning.
- Setiap perubahan kontrak API wajib memperbarui `AGENTS.md`, `docs/api_contract.md`, `src/api/schemas.py`, dan `tests/test_api_contract.py`.
- Semua angka metrik harus berasal dari eksekusi nyata.
