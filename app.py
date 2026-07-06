import os, warnings, io, requests, numpy as np, pandas as pd
import gradio as gr, joblib, tensorflow as tf, PIL.Image
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

warnings.filterwarnings("ignore")

IMG_SIZE = (224, 224)
SAMPLE_N = 8

# ── Load pretrained extractors (no training, ImageNet weights) ─────────────
def build_extractor(name):
    kw = dict(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    if name == "VGG16":
        return VGG16(**kw), vgg_pre
    elif name == "ResNet50":
        return ResNet50(**kw), res_pre
    raise ValueError(name)

models_dict = {}
for mname in ["VGG16", "ResNet50"]:
    m, fn = build_extractor(mname)
    m.trainable = False
    models_dict[mname] = {"model": m, "preprocess": fn}

# ── Load precomputed embedding databases ────────────────────────────────────
databases = {}
for mname in ["VGG16", "ResNet50"]:
    pkl = f"db_{mname.lower()}.pkl"
    if os.path.exists(pkl):
        databases[mname] = joblib.load(pkl)

# ── Load metadata (test-set rows, used only for sample images) ─────────────
shoes_meta = pd.DataFrame()
if os.path.exists("shoes_metadata.csv"):
    shoes_meta = pd.read_csv("shoes_metadata.csv")

# ── Image fetch helper (URL first, CDN fallback, gray placeholder last) ────
def fetch_image(row, size=IMG_SIZE):
    url_col = next((c for c in ["link", "url", "image_url"] if c in row.index and pd.notna(row.get(c))), None)
    try:
        if url_col:
            resp = requests.get(row[url_col], timeout=6)
            resp.raise_for_status()
            return PIL.Image.open(io.BytesIO(resp.content)).convert("RGB").resize(size)
        elif "id" in row.index and pd.notna(row["id"]):
            resp = requests.get(f"https://assets.myntassets.com/v1/images/{int(row['id'])}/1/image.jpg", timeout=6)
            resp.raise_for_status()
            return PIL.Image.open(io.BytesIO(resp.content)).convert("RGB").resize(size)
    except Exception:
        pass
    return PIL.Image.new("RGB", size, (220, 220, 220))

# ── Similarity metric (fixed: was dead code before, only Cosine kept active
#    since that's the only one enabled in the notebook's SIM_FNS) ──────────
def cosine_scores(q, db):
    return sk_cosine(q.reshape(1, -1), db)[0]

SIM_FNS = {
    "Cosine Similarity": cosine_scores,
}

# ── Preprocessing + retrieval ────────────────────────────────────────────────
def prep_query(pil_img, prep_fn, img_size=IMG_SIZE):
    arr = np.array(pil_img.resize(img_size)).astype("float32")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return prep_fn(arr)[np.newaxis]

def retrieve(query_emb, db, sim_fn, top_k=10):
    scores = sim_fn(query_emb, db["embeddings"])
    idx = np.argsort(scores)[::-1][:top_k]
    res = db["metadata"].iloc[idx].copy().reset_index(drop=True)
    res["score"] = scores[idx]
    return res

# ── Sample images (fetched once at startup, shown as a clickable gallery) ──
SAMPLE_IMGS = []
if not shoes_meta.empty:
    sample_rows = shoes_meta.sample(min(SAMPLE_N, len(shoes_meta)), random_state=42)
    for _, row in sample_rows.iterrows():
        img = fetch_image(row)
        label = str(row.get("item_label", row.get("id", f"Sample {len(SAMPLE_IMGS)+1}")))
        SAMPLE_IMGS.append((img, label))

# ── Helper functions for Gradio ─────────────────────────────────────────────
def search_fn(img, model_name, sim_name, top_k):
    if img is None:
        return [], None
    info = models_dict[model_name]
    db = databases.get(model_name)
    if db is None:
        return [], pd.DataFrame({"error": ["Database not loaded"]})

    q_emb = info["model"].predict(prep_query(img, info["preprocess"]), verbose=0)[0]
    res_df = retrieve(q_emb, db, SIM_FNS[sim_name], top_k=int(top_k))

    gallery_items = []
    for i, (_, row) in enumerate(res_df.iterrows()):
        pil = fetch_image(row)
        cap = f"#{i+1}  {row.get('item_label', row.get('id',''))} · {row.get('baseColour','')} ({row['score']:.3f})"
        gallery_items.append((pil, cap))

    cols = [c for c in ["item_label", "baseColour", "gender", "score"] if c in res_df.columns]
    disp_df = res_df[cols].rename(columns={
        "item_label": "Product ID", "baseColour": "Colour", "gender": "Gender", "score": "Score"
    }).copy()
    return gallery_items, disp_df

def clear_results():
    return [], pd.DataFrame()

def load_sample_image(evt: gr.SelectData):
    if evt.index < len(SAMPLE_IMGS):
        pil_img, _ = SAMPLE_IMGS[evt.index]
        return pil_img, [], pd.DataFrame()
    return gr.update(), [], pd.DataFrame()

# ── Theme (Forced Light Matcha Green) ───────────────────────────────────────
light_matcha_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"]
).set(
    body_background_fill="#F4F8F4",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#D5E0D5",
    button_primary_background_fill="#8AAB7F",
    button_primary_background_fill_hover="#729168",
    button_primary_text_color="#FFFFFF"
)

custom_css = """
/* --- PUSATKAN KESELURUHAN HALAMAN --- */
body { background-color: #F4F8F4 !important; }
.gradio-container {
    max-width: 1100px !important;
    margin: 40px auto !important;
}

h1 { text-align:center; font-weight:800 !important; letter-spacing:-0.025em; color:#2C3B2C !important; }
.subtitle p { text-align:center; color:#5D705D !important; font-size:1rem; margin-top:-8px; }
.sample-hint p { text-align:center !important; color:#758A75 !important; font-size:0.9rem; margin:6px 0 8px; font-weight:500; }

.gr-button-primary { border-radius:8px !important; font-weight:600 !important;
                     transition:all 0.2s ease !important; }
.gr-button-primary:hover { transform:translateY(-1px);
    box-shadow:0 6px 12px -2px rgba(138,171,127,0.4) !important; }
.gr-gallery { border-radius:10px !important; }

/* --- CSS UNTUK HORIZONTAL SCROLL SAMPEL GAMBAR (single row, no pagination) --- */
#scroll-examples .grid-wrap {
    overflow: visible !important;
}
#scroll-examples .grid-container,
#scroll-examples .grid-wrap > div {
    display: flex !important;
    flex-wrap: nowrap !important;
    flex-direction: row !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    padding-bottom: 8px;
    gap: 8px;
    justify-content: flex-start;
}
#scroll-examples .grid-container > *,
#scroll-examples .grid-wrap > div > * {
    flex: 0 0 auto !important;
    width: 75px !important;
    height: 75px !important;
}
/* Sembunyikan tombol panah/paginasi bawaan Gradio */
#scroll-examples .paginate,
#scroll-examples .icon-buttons {
    display: none !important;
}
#scroll-examples .grid-container::-webkit-scrollbar,
#scroll-examples .grid-wrap > div::-webkit-scrollbar {
    height: 8px;
}
#scroll-examples .grid-container::-webkit-scrollbar-track,
#scroll-examples .grid-wrap > div::-webkit-scrollbar-track {
    background: #E8F0E8;
    border-radius: 4px;
}
#scroll-examples .grid-container::-webkit-scrollbar-thumb,
#scroll-examples .grid-wrap > div::-webkit-scrollbar-thumb {
    background-color: #8AAB7F;
    border-radius: 4px;
}

/* --- FIX: kunci ukuran thumbnail gallery hasil pencarian (anti-glitch) --- */
#results-gallery .thumbnail-item, #results-gallery button.thumbnail-item {
    width: 100% !important;
    height: 180px !important;
    aspect-ratio: 1 / 1 !important;
    overflow: hidden !important;
}
#results-gallery img { width: 100% !important; height: 100% !important; object-fit: cover !important; }
"""

# Script JavaScript untuk menghapus class 'dark' secara paksa saat web dimuat
force_light_js = """
function() {
    function stripDark() {
        document.body.classList.remove('dark');
        document.documentElement.classList.remove('dark');
    }
    stripDark();

    // Gradio (atau browser lewat prefers-color-scheme) bisa nambahin
    // class 'dark' lagi beberapa saat setelah load pertama.
    // MutationObserver ini terus memantau & langsung menghapusnya lagi
    // setiap kali class itu muncul, bukan cuma sekali di awal.
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            if (m.attributeName === 'class') {
                stripDark();
            }
        });
    });

    observer.observe(document.documentElement, { attributes: true });
    observer.observe(document.body, { attributes: true });
}
"""

with gr.Blocks(title="Casual Shoes Similarity Search", theme=light_matcha_theme, css=custom_css, js=force_light_js) as demo:

    with gr.Column(elem_classes="subtitle"):
        gr.Markdown("""
        # 👟 Casual Shoes Image Similarity Search
        Upload a shoe image or select a sample below · VGG16 & ResNet50 · Cosine Similarity
        """)

    with gr.Row(equal_height=True):

        # Left panel — controls
        with gr.Column(scale=1, min_width=280):
            img_in = gr.Image(type="pil", label="Query Image", height=260)

            # ── Sample Images (horizontal scroll, click to load) ────────────
            if SAMPLE_IMGS:
                with gr.Column(elem_id="scroll-examples"):
                    gr.Markdown("**🖼️ Sample Images** — click one to use as a query", elem_classes="sample-hint")
                    sample_gallery = gr.Gallery(
                        value=SAMPLE_IMGS,
                        columns=max(len(SAMPLE_IMGS), 1),
                        rows=1,
                        height=90,
                        label="",
                        show_label=False,
                        allow_preview=False,
                        object_fit="cover",
                        elem_id="scroll-examples-gallery",
                    )

            with gr.Group():
                m_sel = gr.Dropdown(["VGG16", "ResNet50"], value="VGG16", label="🧠 Feature Extractor")
                s_sel = gr.Dropdown(list(SIM_FNS.keys()), value="Cosine Similarity", label="📐 Similarity Metric")
                k_sl = gr.Slider(5, 10, step=5, value=5, label="Top-K Results")
            btn = gr.Button("🔍  Find Similar Shoes", variant="primary", size="lg")

        # Right panel — results
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Similar Shoes Found",
                columns=5,
                height=260,
                object_fit="cover",
                show_label=True,
                allow_preview=True,
                elem_id="results-gallery",
            )
            tbl = gr.Dataframe(label="Score Details", wrap=True, row_count=(5, "fixed"))

    # ── Event binding ─────────────────────────────────────────────────────────
    if SAMPLE_IMGS:
        sample_gallery.select(fn=load_sample_image, outputs=[img_in, gallery, tbl])

    img_in.upload(fn=clear_results, outputs=[gallery, tbl])
    img_in.clear(fn=clear_results, outputs=[gallery, tbl])

    btn.click(fn=search_fn, inputs=[img_in, m_sel, s_sel, k_sl], outputs=[gallery, tbl])

if __name__ == "__main__":
    demo.launch()