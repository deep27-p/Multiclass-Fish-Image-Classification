import os
import warnings
import logging

# -----------------------------
# SUPPRESS ALL WARNINGS BEFORE IMPORTING TENSORFLOW
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
warnings.filterwarnings("ignore")  # Ignore Python warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model_path = r"C:\Users\sowmi\OneDrive\Desktop\python\multiclass Fish\images.cv_jzk6llhf18tm3k0kyttxz\best_fish_model.keras"
model = tf.keras.models.load_model(model_path, compile=False)

# -----------------------------
# CLASS NAMES
# -----------------------------
class_names = [
    'animal_fish', 'animal_fish_bass', 'fish_sea_food_black_sea_sprat',
    'fish_sea_food_gilt_head_bream', 'fish_sea_food_hourse_mackerel',
    'fish_sea_food_red_mullet', 'fish_sea_food_red_sea_bream',
    'fish_sea_food_sea_bass', 'fish_sea_food_shrimp',
    'fish_sea_food_striped_red_mullet', 'fish_sea_food_trout'
]

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered", page_icon="🐠")
st.title("🐠 Fish Classification App")
st.markdown("Upload an image of a fish, and the model will predict its category with **confidence visualization** 📊")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(image):
    img = image.resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    preds = model.predict(img_array, verbose=0)
    preds = np.array(preds)

    # Multi-class
    if preds.ndim == 2 and preds.shape[1] > 1:
        probs = tf.nn.softmax(preds[0]).numpy()
        idx = int(np.argmax(probs))
        predicted_class = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        confidence = 100.0 * float(probs[idx])
        return predicted_class, confidence, probs

    # Single-output / binary
    if (preds.ndim == 2 and preds.shape[1] == 1) or (preds.ndim == 1 and preds.size == 1):
        val = float(np.squeeze(preds))
        p = float(tf.sigmoid(val).numpy())
        probs = np.array([1.0 - p, p], dtype=float)
        predicted_class = class_names[1] if p >= 0.5 else class_names[0]
        confidence = 100.0 * max(p, 1.0 - p)
        return predicted_class, confidence, probs

    # Fallback for unexpected shapes
    flat = preds.ravel()
    if flat.size > 1:
        probs = tf.nn.softmax(flat).numpy()
        idx = int(np.argmax(probs))
        predicted_class = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        confidence = 100.0 * float(probs[idx])
        return predicted_class, confidence, probs

    # Final fallback: single scalar
    val = float(flat[0]) if flat.size == 1 else 0.0
    p = float(tf.sigmoid(val).numpy())
    probs = np.array([1.0 - p, p], dtype=float)
    predicted_class = class_names[0] if len(class_names) > 0 else "class_0"
    confidence = 100.0 * max(p, 1.0 - p)
    return predicted_class, confidence, probs

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 Uploaded Image', width='stretch')

    st.markdown("### 🧠 Classifying...")
    predicted_class, confidence, scores = predict(image)

    # Display results
    st.success(f"✅ **Predicted Category:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")

    # Prepare data for Plotly
    scores = np.array(scores, dtype=float)
    names = class_names if len(scores) == len(class_names) else [f"class_{i}" for i in range(len(scores))]

    df = pd.DataFrame({
        'Fish Category': names,
        'Confidence (%)': [float(s) * 100 for s in scores]
    }).sort_values('Confidence (%)', ascending=True)

    # Plotly bar chart with proper config to avoid warnings
    fig = px.bar(
        df,
        x='Confidence (%)',
        y='Fish Category',
        orientation='h',
        color='Confidence (%)',
        color_continuous_scale='teal',
        text_auto='.2f',
        title='Model Confidence per Class'
    )

    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13),
    )

    # Streamlit plotly_chart with config
    st.plotly_chart(fig, config={"displayModeBar": True}, use_container_width=True)
    st.caption("💡 Tip: The longer the bar, the more confident the model is in that class.")
else:
    st.info("👆 Upload a fish image (JPG/PNG) to begin classification.")
