# 👟 Casual Shoes Image Similarity Search

Sistem pencarian gambar sepatu kasual (*image similarity search*) berbasis **pretrained CNN feature extractor** — tanpa training ulang. Diberikan satu gambar sepatu sebagai *query*, sistem akan mengembalikan sepatu-sepatu lain yang paling mirip dari database.

## 🧠 Arsitektur

- **Feature Extractor**: `VGG16` & `ResNet50` (pretrained ImageNet, `include_top=False`, `pooling='avg'`) — dipakai langsung sebagai *embedding extractor* tanpa proses training.
- **Dataset**: [`paramaggarwal/fashion-product-images-dataset`](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (Kaggle), difilter hanya kategori `articleType == 'Casual Shoes'`.
- **Similarity Metric**: Cosine Similarity antar vektor embedding.

## 🔄 Alur Pipeline

| # | Tahap | Keterangan |
|---|---|---|
| 1 | Data Collection | Download & merge `styles.csv` + `images.csv` |
| 2 | Filter Casual Shoes | `articleType == 'Casual Shoes'` |
| 3 | Data Splitting | Train / Validation / Test (80/10/10) |
| 4 | Feature Extraction | Embedding dari VGG16 & ResNet50 |
| 5 | Similarity Search | Cosine similarity, top-K retrieval |
| 6 | Visualisasi | Grid gambar query vs hasil top-5 |
| 7 | Evaluasi | Mean Reciprocal Rank (MRR) & Recall@K dengan anotasi manusia |
| 8 | Demo Interaktif | Gradio App (3 tab: Search, Bandingkan Metric, Hasil Evaluasi) |
| 9 | Deployment | Export siap-pakai ke Hugging Face Spaces |

## 📊 Evaluasi

Kualitas hasil retrieval diukur menggunakan:
- **Mean Reciprocal Rank (MRR)** — dihitung dari anotasi manual (klik gambar paling relevan), disimpan otomatis ke Excel (`mrr_annotations.xlsx`).
- **Recall@5** dan **Recall@10** — berdasarkan relevansi kombinasi `gender` + `baseColour`.

Hasil evaluasi divisualisasikan dalam bentuk bar chart perbandingan antar kombinasi model × metric.

## 🖥️ Demo Interaktif (Gradio)

Aplikasi Gradio menyediakan 3 tab:
1. **🔍 Search** — Upload gambar sepatu → hasil top-K lengkap dengan gambar, label, warna, dan skor similarity.
2. **📊 Bandingkan Metric** — Menampilkan top-5 hasil dari semua metric similarity untuk satu query sekaligus.
3. **📈 Hasil Evaluasi** — Tabel dan chart perbandingan performa semua kombinasi model.

## 🚀 Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>
```

### 2. Siapkan data
Notebook mendukung dua mode:
- **Mode ZIP lokal**: letakkan `casual_shoes_data.csv` dan `casual_shoes_images_dataset.zip` di direktori kerja.
- **Mode Kaggle**: gunakan `kagglehub` untuk mengunduh dataset asli secara langsung (kode tersedia namun di-*comment*, aktifkan sesuai kebutuhan).

### 3. Install dependencies
```bash
pip install kagglehub gradio tensorflow seaborn openpyxl ipywidgets pandas numpy scikit-learn pillow matplotlib
```

### 4. Jalankan notebook
Buka dan jalankan `Casual_Shoes_Image_Similarity_v8.ipynb` di Jupyter Notebook, Google Colab, atau lingkungan sejenis secara berurutan dari atas ke bawah.

## 📦 Deploy ke Hugging Face Spaces

Notebook menyediakan cell otomatis untuk mengemas semua *artifact* yang dibutuhkan (`app.py`, `requirements.txt`, embedding `.pkl`, gambar evaluasi, dsb.) ke dalam satu file ZIP siap unggah:

1. Jalankan cell **"Export ke Hugging Face Spaces"** di notebook — akan menghasilkan `casual_shoes_hf_space.zip`.
2. Buka [huggingface.co/new-space](https://huggingface.co/new-space), pilih **SDK: Gradio**, lalu buat Space baru.
3. Extract isi ZIP tersebut dan upload semua filenya ke tab **Files** pada Space.
4. Commit — Space akan otomatis build dan berjalan.

## 📁 Struktur Output Utama

```
casual_shoes_artifacts/
├── db_vgg16.pkl              # Database embedding VGG16
├── db_resnet50.pkl           # Database embedding ResNet50
├── casual_shoes_preview.png  # Preview sampel dataset
├── eval_comparison.png       # Chart perbandingan evaluasi
├── eval_heatmap.png          # Heatmap hasil evaluasi
└── sim_*.png                 # Visualisasi hasil similarity per model
```

## 🛠️ Tech Stack

- Python, TensorFlow/Keras (VGG16, ResNet50)
- Pandas, NumPy, scikit-learn
- Gradio (demo interaktif)
- Matplotlib, Seaborn (visualisasi)
- ipywidgets, openpyxl (anotasi & evaluasi MRR)

## 📄 Lisensi

Sesuaikan dengan kebutuhanmu, misalnya [MIT License](https://opensource.org/licenses/MIT).

---

> Dataset yang digunakan mengikuti lisensi dari sumber aslinya di Kaggle: [paramaggarwal/fashion-product-images-dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).
