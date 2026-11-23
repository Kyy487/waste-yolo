# Kontrak API v1 — Referensi

Dokumen ini adalah ringkasan dari `AGENTS.md` bagian 8. Sumber kebenaran tetap [`AGENTS.md`](../AGENTS.md). Perubahan kontrak wajib memperbarui `AGENTS.md`, dokumen ini, `src/api/schemas.py`, dan `tests/test_api_contract.py`. Perubahan yang merusak mewajibkan prefix `/v2`.

## Informasi dasar

| Item | Nilai |
| --- | --- |
| Base URL lokal | `http://localhost:8000` |
| Prefix versi | `/v1` |
| Format | JSON (`application/json`), kecuali upload gambar (`multipart/form-data`) |
| Encoding | UTF-8 |
| Autentikasi | Tidak ada untuk lokal/demo |
| Dokumentasi otomatis | `/docs` dan `/openapi.json` |

## Endpoint

| Method | Path | Fungsi |
| --- | --- | --- |
| `GET` | `/v1/health` | Cek layanan dan status model |
| `GET` | `/v1/classes` | Daftar 9 kelas beserta indeks |
| `GET` | `/v1/model` | Metadata model aktif |
| `POST` | `/v1/predict` | Prediksi satu gambar |
| `POST` | `/v1/predict/batch` | Prediksi 1–16 gambar |

## `GET /v1/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "api_version": "v1"
}
```

## `GET /v1/classes`

Urutan indeks tetap mengikuti urutan abjad nama folder YOLO-cls.

```json
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
```

## `GET /v1/model`

```json
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
```

Nilai `metrics` diisi dari evaluasi nyata dan tetap `null` sebelum evaluasi selesai.

## `POST /v1/predict`

Request: `multipart/form-data`, field `file` berisi JPG/PNG maksimal 5 MB. Query opsional `top_k` bertipe integer `1–9`, default `3`.

```json
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
```

## `POST /v1/predict/batch`

Request: `multipart/form-data`, field `files` berisi 1–16 gambar. Jika satu file gagal, file itu mengembalikan objek error pada posisinya dan file lain tetap diproses.

```json
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
```

## Format error seragam

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "File bukan gambar yang valid.",
    "details": null
  }
}
```

| HTTP | Code | Kapan terjadi |
| --- | --- | --- |
| 400 | `INVALID_IMAGE` | File rusak atau bukan gambar |
| 400 | `INVALID_PARAMETER` | `top_k` di luar 1–9, dll. |
| 413 | `FILE_TOO_LARGE` | Ukuran > 5 MB |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Format selain JPG/PNG |
| 422 | `VALIDATION_ERROR` | Field wajib tidak ada |
| 429 | `RATE_LIMITED` | Terlalu banyak permintaan, jika diaktifkan |
| 500 | `INTERNAL_ERROR` | Kesalahan tak terduga, tanpa stack trace |
| 503 | `MODEL_NOT_LOADED` | Model belum siap |

## Aturan perilaku

- `confidence` bertipe float `0–1`; total probabilitas seluruh kelas ≈ 1 dengan toleransi ±0,001.
- `top_k` diurutkan menurun berdasarkan confidence.
- Praproses API identik dengan training: resize 224×224 dengan padding, RGB.
- Nama field memakai `snake_case`; nama kelas huruf kecil bahasa Inggris persis seperti daftar di atas.
- Model dimuat sekali saat startup, bukan per request.
- Setiap request dicatat: waktu, endpoint, status, durasi. Gambar pengguna tidak disimpan.

## Skema Pydantic acuan

```python
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
```

## Versioning

- Perubahan non-merusak, misalnya menambah field opsional, boleh di `v1` setelah disetujui Project Lead.
- Perubahan merusak, misalnya menghapus/mengubah nama field, mengubah makna, atau mengubah urutan kelas, mewajibkan `/v2` dan dokumentasi migrasi.
