"""
Streamlit Web UI for Fashion MNIST HOG + SVM Classifier.

Run with:   streamlit run app.py
Requires:   pip install streamlit

This file is fully additive — no existing files were modified.
"""

import streamlit as st
import cv2
import numpy as np

# ── Imports from the existing src package ──────────────────────────────────
from src.model import load_model, model_exists
from src.preprocessing import preprocess_data
from src.features import extract_hog_features


# ── Class names (duplicated from main.py) ──────────────────────────────────
# main.py runs dataset-loading side effects at import time, so importing the
# list from there would trigger the entire pipeline.  A small duplication is
# the safest approach.
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Emoji mapping for each class — gives the result card a visual identity
CLASS_ICONS = {
    "T-shirt/top": "👕", "Trouser": "👖", "Pullover": "🧶",
    "Dress": "👗",       "Coat": "🧥",    "Sandal": "🩴",
    "Shirt": "👔",       "Sneaker": "👟",  "Bag": "👜",
    "Ankle boot": "🥾",
}


# ── Model loading (cached once per Streamlit session) ──────────────────────
@st.cache_resource
def get_model():
    """Load the trained SVM from disk exactly once."""
    if not model_exists():
        return None
    return load_model()


# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="centered",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — all visual overrides live here
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global overrides ────────────────────────────────────────────────────── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Subtle animated gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1333 40%, #24243e 100%);
    background-attachment: fixed;
}

/* Remove default Streamlit header and footer */
header[data-testid="stHeader"] {
    background: transparent !important;
}
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── Hero section ────────────────────────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    margin-bottom: 0.5rem;
}
.hero-icon {
    font-size: 3.8rem;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
    margin-bottom: 0.2rem;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-10px); }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin: 0.3rem 0 0.5rem 0;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #a5a3b5;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.55;
}

/* ── Status badge (model loaded) ─────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #34d399;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    margin: 0.8rem auto 0 auto;
}

/* ── Glass card ──────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
}

/* ── Upload zone styling ─────────────────────────────────────────────────── */
.upload-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e0f0;
    margin-bottom: 0.3rem;
}
.upload-hint {
    font-size: 0.82rem;
    color: #7c7a8e;
    margin-bottom: 1rem;
}

/* Style Streamlit's file-uploader widget */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(129, 140, 248, 0.35) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(129, 140, 248, 0.65) !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.12);
}

/* ── Prediction result card ──────────────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(129, 140, 248, 0.2);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
}
.result-icon {
    font-size: 3.2rem;
    margin-bottom: 0.4rem;
}
.result-label-header {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #9b99b0;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.result-class-name {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e0f0;
    margin-bottom: 0.6rem;
}
.result-pill {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
}

/* ── Confidence unavailable note ─────────────────────────────────────────── */
.confidence-note {
    text-align: center;
    font-size: 0.78rem;
    color: #6b6980;
    margin-top: 0.6rem;
    font-style: italic;
}

/* ── Image container ─────────────────────────────────────────────────────── */
.image-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 0.8rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.image-card img {
    border-radius: 10px;
}
.image-label {
    text-align: center;
    font-size: 0.78rem;
    color: #7c7a8e;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* ── Error card ──────────────────────────────────────────────────────────── */
.error-card {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    color: #fca5a5;
    font-size: 0.95rem;
    line-height: 1.6;
}
.error-card code {
    background: rgba(239, 68, 68, 0.15);
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-size: 0.85rem;
}

/* ── Footer ──────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    font-size: 0.75rem;
    color: #4a4860;
    letter-spacing: 0.04em;
}
.app-footer a {
    color: #818cf8;
    text-decoration: none;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(129, 140, 248, 0.25), transparent);
    border: none;
    margin: 1.5rem 0;
}

/* ── Hide default streamlit elements in styled sections ──────────────────── */
.block-container {
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-icon">👗</div>
    <div class="hero-title">Fashion MNIST Classifier</div>
    <div class="hero-subtitle">
        Upload a clothing or accessory image and our HOG + SVM model
        will instantly identify its category.
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — logic identical to original
# ══════════════════════════════════════════════════════════════════════════════
model = get_model()

if model is None:
    st.markdown("""
    <div class="error-card">
        ⚠️ <strong>No trained model found!</strong><br><br>
        Run <code>python main.py</code> first to train and save the SVM model,
        then restart this app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Model-loaded status badge
st.markdown("""
<div style="text-align:center;">
    <span class="status-badge">● &nbsp;Model loaded successfully</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="glass-card">
    <div class="upload-header">📤 &nbsp;Upload an Image</div>
    <div class="upload-hint">Drag & drop or click below · PNG, JPG, JPEG accepted</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image…",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    # ── Decode the uploaded file ───────────────────────────────────────────
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.markdown("""
        <div class="error-card">
            Could not decode the uploaded image — please try a different file.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Preprocessing (mirrors predict_custom_image in src/predict.py) ─────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    resized = resized.astype("float32") / 255.0
    resized = resized.reshape(1, 28, 28)

    processed, _ = preprocess_data(resized, resized)

    # ── HOG feature extraction ─────────────────────────────────────────────
    hog = extract_hog_features(processed)

    # ── Prediction ─────────────────────────────────────────────────────────
    with st.spinner(""):
        prediction = model.predict(hog)[0]
        predicted_class = CLASS_NAMES[prediction]

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Two-column layout: image on left, result on right ──────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        # Show the uploaded image (BGR → RGB for correct colours)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(image_rgb, use_container_width=True)
        st.markdown('<div class="image-label">Uploaded Image</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        icon = CLASS_ICONS.get(predicted_class, "🏷️")
        st.markdown(f"""
        <div class="result-card">
            <div class="result-icon">{icon}</div>
            <div class="result-label-header">Predicted Category</div>
            <div class="result-class-name">{predicted_class}</div>
            <div class="result-pill">Class {prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence / probability (if available) ────────────────────────
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(hog)[0]
                st.write("**Class probabilities:**")
                prob_dict = {CLASS_NAMES[i]: f"{p:.2%}" for i, p in enumerate(probas)}
                st.json(prob_dict)
            except Exception:
                pass   # predict_proba not available (probability=False) — skip silently
        else:
            st.markdown(
                '<div class="confidence-note">'
                'Confidence scores unavailable — model was not trained with '
                '<code>probability=True</code></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="styled-divider"></div>
<div class="app-footer">
    Powered by <strong>HOG + SVM</strong> · scikit-learn &nbsp;·&nbsp;
    Built with <a href="https://streamlit.io" target="_blank">Streamlit</a>
</div>
""", unsafe_allow_html=True)
