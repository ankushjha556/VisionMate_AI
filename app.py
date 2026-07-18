"""
VisionMate AI — Streamlit Demo
Live demo of Modules 1 (Scene Understanding), 2 (OCR), and 3 (Currency Detection).

Run with: streamlit run app.py
"""

import io
import os
from collections import defaultdict

import numpy as np
import streamlit as st
from PIL import Image
from gtts import gTTS

st.set_page_config(page_title="VisionMate AI — Demo", page_icon="👁️", layout="centered")

# ------------------------------------------------------------------
# Cached model loaders — each model loads once per session, not per interaction
# ------------------------------------------------------------------

@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolo11n.pt")


@st.cache_resource
def load_ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


@st.cache_resource
def load_currency_model():
    model_path = "currency_classifier.keras"
    if not os.path.exists(model_path):
        return None
    from tensorflow.keras.models import load_model
    return load_model(model_path)


CURRENCY_LABELS = [
    "1Hundrednote", "2Hundrednote", "2Thousandnote", "5Hundrednote",
    "Fiftynote", "Tennote", "Twentynote",
]
DENOMINATION_MAP = {
    "Tennote": "10 rupees", "Twentynote": "20 rupees", "Fiftynote": "50 rupees",
    "1Hundrednote": "100 rupees", "2Hundrednote": "200 rupees",
    "5Hundrednote": "500 rupees", "2Thousandnote": "2000 rupees",
}

# ------------------------------------------------------------------
# Shared reasoning helpers (same logic as the master notebook)
# ------------------------------------------------------------------

def get_spatial_zone(x_center, image_width):
    if x_center < image_width * 0.33:
        return "left"
    elif x_center > image_width * 0.66:
        return "right"
    return "ahead"


def reason_scene(results, image_width, min_confidence=0.35):
    boxes = results[0].boxes
    filtered = [b for b in boxes if float(b.conf[0]) >= min_confidence]

    zone_groups = defaultdict(int)
    for box in filtered:
        cls_name = results[0].names[int(box.cls[0])]
        x_center = box.xywh[0][0].item()
        zone_groups[(cls_name, get_spatial_zone(x_center, image_width))] += 1

    if not zone_groups:
        return "No significant objects detected around you."

    priority_classes = {"person", "car", "bus", "truck", "motorcycle", "bicycle"}

    def sort_key(item):
        (cls_name, zone), count = item
        return (0 if cls_name in priority_classes else 1, -count)

    sentences = []
    for (cls_name, zone), count in sorted(zone_groups.items(), key=sort_key):
        obj_phrase = f"a {cls_name}" if count == 1 else f"{count} {cls_name}s"
        sentences.append(f"{obj_phrase} {zone}")

    if len(sentences) == 1:
        return f"There is {sentences[0]}."
    return "There is " + ", ".join(sentences[:-1]) + f", and {sentences[-1]}."


def reason_ocr(ocr_results, min_confidence=0.5):
    sorted_results = sorted(ocr_results, key=lambda r: r[0][0][1])
    confident_lines, skipped = [], 0
    for (_bbox, text, confidence) in sorted_results:
        if confidence >= min_confidence:
            confident_lines.append(text.strip())
        else:
            skipped += 1
    if not confident_lines:
        return "No readable text detected.", skipped
    return ". ".join(confident_lines), skipped


def correct_ocr_text(text):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    words = text.split()
    corrected, corrections_made = [], []
    for word in words:
        clean = word.strip(".,!?-")
        if clean.isdigit() or len(clean) <= 2 or clean.isupper():
            corrected.append(word)
            continue
        if clean.lower() not in spell:
            fix = spell.correction(clean)
            if fix and fix != clean.lower():
                corrections_made.append((clean, fix))
                fixed_word = fix.capitalize() if clean[0].isupper() else fix
                corrected.append(word.replace(clean, fixed_word))
                continue
        corrected.append(word)
    return " ".join(corrected), corrections_made


def predict_currency(model_currency, img: Image.Image, confidence_threshold=0.5):
    from tensorflow.keras.preprocessing import image as keras_image

    img_resized = img.convert("RGB").resize((224, 224))
    arr = np.expand_dims(keras_image.img_to_array(img_resized) / 255.0, axis=0)
    preds = model_currency.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])
    denom = DENOMINATION_MAP.get(CURRENCY_LABELS[idx], CURRENCY_LABELS[idx])

    if confidence >= confidence_threshold:
        spoken = f"This is {denom}."
    else:
        spoken = f"I'm not fully sure, but this might be {denom}. Please verify."
    return spoken, confidence, denom


def speak(text: str) -> bytes:
    """Generate speech audio in-memory (no disk write needed for the demo)."""
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.title("👁️ VisionMate AI")
st.caption(
    "An AI-powered assistive companion for visually impaired users. "
    "Upload an image to try Scene Understanding, Text Reading, and Currency Detection."
)

tab1, tab2, tab3 = st.tabs(["🖼️ Scene Understanding", "📝 Text Reading (OCR)", "💵 Currency Detection"])

# --- Tab 1: Scene Understanding ---
with tab1:
    st.subheader("Module 1 — Scene Understanding")
    st.write("Detects objects and describes their position (left / ahead / right).")
    uploaded_scene = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="scene")

    if uploaded_scene:
        image = Image.open(uploaded_scene).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Detecting objects..."):
            yolo_model = load_yolo()
            temp_path = "temp_scene.jpg"
            image.save(temp_path)
            results = yolo_model(temp_path, conf=0.25)
            image_width = results[0].orig_shape[1]
            annotated = results[0].plot()[:, :, ::-1]

        st.image(annotated, caption="Detected objects", use_container_width=True)

        description = reason_scene(results, image_width)
        st.success(f"**Spoken description:** {description}")
        st.audio(speak(description), format="audio/mp3")

# --- Tab 2: OCR ---
with tab2:
    st.subheader("Module 2 — Text Reading")
    st.write("Reads signs, labels, or documents aloud, with automatic misread correction.")
    uploaded_text = st.file_uploader("Upload a photo containing text", type=["jpg", "jpeg", "png"], key="ocr")

    if uploaded_text:
        image = Image.open(uploaded_text).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Reading text..."):
            reader = load_ocr_reader()
            temp_path = "temp_ocr.jpg"
            image.save(temp_path)
            ocr_results = reader.readtext(temp_path)
            raw_text, skipped = reason_ocr(ocr_results, min_confidence=0.5)
            corrected_text, corrections = correct_ocr_text(raw_text)

        if corrections:
            st.info(f"Auto-corrected: {', '.join(f'{a} → {b}' for a, b in corrections)}")
        st.success(f"**Spoken text:** {corrected_text}")
        st.audio(speak(corrected_text), format="audio/mp3")

# --- Tab 3: Currency Detection ---
with tab3:
    st.subheader("Module 3 — Currency Detection")
    st.write("Identifies Indian currency notes (₹10–₹2000).")

    model_currency = load_currency_model()
    if model_currency is None:
        st.warning(
            "`currency_classifier.keras` not found in this folder. "
            "Train it using the master notebook, then place the file next to `app.py` to enable this tab."
        )
    else:
        uploaded_currency = st.file_uploader(
            "Upload a photo of a currency note", type=["jpg", "jpeg", "png"], key="currency"
        )
        if uploaded_currency:
            image = Image.open(uploaded_currency).convert("RGB")
            st.image(image, caption="Uploaded note", use_container_width=True)

            with st.spinner("Identifying denomination..."):
                spoken, confidence, denom = predict_currency(model_currency, image)

            st.success(f"**Prediction:** {denom} (confidence: {confidence:.2f})")
            st.write(f"**Spoken result:** {spoken}")
            st.audio(speak(spoken), format="audio/mp3")

st.divider()
st.caption(
    "VisionMate AI — built with YOLOv11n, EasyOCR, and a fine-tuned MobileNetV2. "
    "See the [master notebook](../notebooks/VisionMate_Master.ipynb) for the full pipeline, "
    "including Obstacle Warning and Multilingual Voice."
)
