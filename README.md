# 👟 Casual Shoes Image Similarity Search

An image similarity search system for casual shoes, built on **pretrained CNN feature extractors** — no retraining involved. Given a query shoe image, the system retrieves the most visually similar shoes from a database.

## 🧠 Architecture

- **Feature Extractor**: `VGG16` & `ResNet50` (pretrained on ImageNet, `include_top=False`, `pooling='avg'`) — used directly as embedding extractors with no additional training.
- **Dataset**: [`paramaggarwal/fashion-product-images-dataset`](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (Kaggle), filtered to only `articleType == 'Casual Shoes'`.
- **Similarity Metric**: Cosine similarity between embedding vectors.

## 🔄 Pipeline Overview

| # | Stage | Description |
|---|---|---|
| 1 | Data Collection | Download & merge `styles.csv` + `images.csv` |
| 2 | Filter Casual Shoes | `articleType == 'Casual Shoes'` |
| 3 | Data Splitting | Train / Validation / Test (80/10/10) |
| 4 | Feature Extraction | Embeddings from VGG16 & ResNet50 |
| 5 | Similarity Search | Cosine similarity, top-K retrieval |
| 6 | Visualization | Grid of query vs. top-5 results |
| 7 | Evaluation | Mean Reciprocal Rank (MRR) & Recall@K with human annotation |
| 8 | Interactive Demo | Gradio app (3 tabs: Search, Compare Metrics, Evaluation Results) |
| 9 | Deployment | Ready-to-upload export to Hugging Face Spaces |

## 📊 Evaluation

Retrieval quality is measured using:
- **Mean Reciprocal Rank (MRR)** — computed from manual annotations (clicking the most relevant image), auto-saved to Excel (`mrr_annotations.xlsx`).
- **Recall@5** and **Recall@10** — based on relevance defined by matching `gender` + `baseColour`.

Evaluation results are visualized as bar charts comparing model × metric combinations.

## 🖥️ Interactive Demo (Gradio)

The Gradio app provides 3 tabs:
1. **🔍 Search** — Upload a shoe image → get top-K results complete with images, labels, colors, and similarity scores.
2. **📊 Compare Metrics** — Displays top-5 results across all similarity metrics for a single query at once.
3. **📈 Evaluation Results** — Table and chart comparing the performance of all model combinations.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### 2. Prepare the data
The notebook supports two modes:
- **Local ZIP mode**: place `casual_shoes_data.csv` and `casual_shoes_images_dataset.zip` in the working directory.
- **Kaggle mode**: use `kagglehub` to download the original dataset directly (code is included but commented out — enable it as needed).

### 3. Install dependencies
```bash
pip install kagglehub gradio tensorflow seaborn openpyxl ipywidgets pandas numpy scikit-learn pillow matplotlib
```

### 4. Run the notebook
Open and run `Casual_Shoes_Image_Similarity_v8.ipynb` in Jupyter Notebook, Google Colab, or a similar environment, executing the cells in order from top to bottom.

## 📦 Deploying to Hugging Face Spaces

The notebook includes a cell that automatically packages all required artifacts (`app.py`, `requirements.txt`, embedding `.pkl` files, evaluation charts, etc.) into a single ZIP file ready for upload:

1. Run the **"Export to Hugging Face Spaces"** cell in the notebook — this generates `casual_shoes_hf_space.zip`.
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space), select **SDK: Gradio**, and create a new Space.
3. Extract the ZIP file and upload all its contents to the **Files** tab of the Space.
4. Commit the changes — the Space will automatically build and start running.

## 📁 Main Output Structure

```
casual_shoes_artifacts/
├── db_vgg16.pkl              # VGG16 embedding database
├── db_resnet50.pkl           # ResNet50 embedding database
├── casual_shoes_preview.png  # Dataset sample preview
├── eval_comparison.png       # Evaluation comparison chart
├── eval_heatmap.png          # Evaluation heatmap
└── sim_*.png                 # Per-model similarity visualization
```

## 🛠️ Tech Stack

- Python, TensorFlow/Keras (VGG16, ResNet50)
- Pandas, NumPy, scikit-learn
- Gradio (interactive demo)
- Matplotlib, Seaborn (visualization)
- ipywidgets, openpyxl (MRR annotation & evaluation)

## 📄 License

Adjust as needed, e.g. [MIT License](https://opensource.org/licenses/MIT).

---

> The dataset used follows the original license from its source on Kaggle: [paramaggarwal/fashion-product-images-dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).
