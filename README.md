# Asisten Hukum Ketenagakerjaan AI

Proyek ini merupakan sistem tanya-jawab berbasis AI untuk domain hukum ketenagakerjaan Indonesia (lembur, PHK, pesangon, alih daya/outsourcing, upah, dll), dibangun dengan melatih model bahasa kecil (Qwen2.5-1.5B) secara bertahap: mulai dari penyesuaian gaya bahasa, penguatan kemampuan reasoning lewat reinforcement learning, hingga integrasi dengan sistem retrieval dokumen hukum (RAG). Tujuannya membuktikan bahwa model kecil, kalau dipadukan dengan pipeline retrieval yang tepat, bisa menjawab pertanyaan hukum secara akurat, transparan (ada proses berpikirnya), dan bisa diverifikasi (ada sitasi sumber), tanpa perlu model raksasa. Jawaban didasarkan pada UU Cipta Kerja beserta PP turunannya; kalau pertanyaan di luar cakupan dokumen, sistem otomatis beralih mencari ke internet.

Contoh:

> **Q:** Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?
>
> **A:** *(model mikir dulu di `<think>...</think>`, lalu jawab)*: berhak, sesuai PP 35/2021 pasal soal upah kerja lembur...
>
> **Sumber:** PP No. 35/2021 (PKWT, Alih Daya, Waktu Kerja dan Upah Lembur), Halaman 18

## Cara Kerja

Model kecil (Qwen2.5-1.5B) dilatih bertahap biar bisa reasoning dan jawab akurat pakai dokumen hukum sebagai konteks, bukan hafalan:

```
Fine-tuning (gaya bahasa & instruksi)
        ↓
GRPO (dilatih reasoning eksplisit + jawaban akurat & berbahasa Indonesia)
        ↓
RAG (dikasih akses ke dokumen hukum + fallback internet)
```

Semua model open-weight, dihosting di HuggingFace, jalan di GPU tunggal (T4).

## Fitur

- **Reasoning transparan**: model nulis proses berpikir (`<think>...</think>`) sebelum jawaban final, bukan cuma nembak jawaban.
- **Sitasi otomatis**: tiap jawaban nunjukin dokumen & halaman sumbernya, bisa dicek manual.
- **Retrieval hybrid**: gabung pencarian keyword (BM25) & makna (embedding semantik) biar nggak miss istilah hukum spesifik maupun pertanyaan yang diparafrase.
- **Query expansion (HyDE)**: pertanyaan pendek "dijawab dulu" secara hipotetis sebelum dicari, biar hasil retrieval lebih nyambung ke jawaban yang benar.
- **Reranking**: dokumen hasil pencarian disortir ulang pakai cross-encoder biar yang paling relevan naik ke atas.
- **Fallback pencarian internet**: kalau dokumen lokal nggak relevan (skor rendah), otomatis pindah cari ke DuckDuckGo.
- **Web UI**: antarmuka chat sederhana pakai Gradio, tinggal buka link dan tanya.

## Arsitektur RAG

```
Pertanyaan user
   → HyDE (generate 2 jawaban hipotetis)
   → Ensemble Retriever (BM25 40% + FAISS 60%) → ambil kandidat dokumen
   → Parent-Child lookup (chunk kecil buat cari, chunk besar buat konteks)
   → Reranker (cross-encoder) → ambil top-3 + skor relevansi
   → skor rendah? → DuckDuckGo Search sebagai pengganti konteks
   → skor cukup? → pakai dokumen lokal sebagai konteks
   → Model GRPO generate jawaban (+ reasoning) dari konteks
   → tempel sitasi sumber
```

Dokumen basis, total sekitar 1.900 halaman, di-index pakai embedding multibahasa dan disimpan di FAISS lokal:

| Dokumen | Isi |
|---|---|
| UU Nomor 6 Tahun 2023 | Cipta Kerja |
| PP Nomor 35 Tahun 2021 | PKWT, Alih Daya, Waktu Kerja, dan Upah Lembur |
| PP Nomor 5 Tahun 2021 | Penyelenggaraan Perizinan Berusaha Berbasis Risiko |
| PP Nomor 51 Tahun 2023 | Pengupahan |

## Struktur Proyek

| File | Isi |
|---|---|
| `Fine-tuning_submission_PGABL_Angga-Yulian-Adi-Pradana.ipynb` | Melatih model dasar biar ngerti instruksi & format chat Bahasa Indonesia |
| `GRPO_submission_PGABL_Angga-Yulian-Adi-Pradana.ipynb` | Melatih model reasoning + akurasi jawaban pakai reinforcement learning |
| `RAG_submission_PGABL_Angga-Yulian-Adi-Pradana.ipynb` | Pipeline retrieval + generation + UI chat |
| `link_huggingface.txt` | Link model hasil training |
| `requirements.txt` | Dependency Python |

## Model

| Tahap | Model |
|---|---|
| Fine-tuned base | [anggapradana/qwen2-5-1-5b-alpaca-id](https://huggingface.co/anggapradana/qwen2-5-1-5b-alpaca-id) |
| GRPO (dipakai RAG) | [anggapradana/qwen2-5-1-5b-grpo-id](https://huggingface.co/anggapradana/qwen2-5-1-5b-grpo-id) |

## Menjalankan

```bash
pip install -r requirements.txt
```

Jalankan notebook urut: Fine-tuning → GRPO → RAG (masing-masing sudah standalone, load model dari HuggingFace di notebook berikutnya). Butuh GPU (dites di Kaggle T4) dan `HF_TOKEN` buat akses/push model ke HuggingFace.

Notebook RAG di bagian akhir buka UI Gradio dengan link publik, tinggal ketik pertanyaan di sana.

## Kenapa model kecil bisa akurat

Qwen2.5-1.5B nggak dihafalin pasal, dia dilatih buat *reasoning* dan *mengikuti konteks*. Jawaban akurat datang dari RAG (dokumen asli yang di-retrieve), bukan dari parameter model. Ini bikin sistem lebih murah dijalankan, jawabannya bisa diverifikasi (ada sitasi), dan gampang di-update, tinggal ganti dokumen PDF-nya, nggak perlu re-training.
