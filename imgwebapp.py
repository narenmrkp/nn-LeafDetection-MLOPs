# imgwebapp.py
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image

import tensorflow as tf
import streamlit as st

# --- UI config ---
st.set_page_config(page_title="Bean Classifier", layout="centered")
st.title("Bean Image Classifier")
st.write("Provide an image URL; app will return a single predicted label + confidence.")

# --- caching decorator compatibility ---
if hasattr(st, "cache_resource"):
    cache_decorator = st.cache_resource
else:
    def cache_decorator(func=None, **kwargs):
        if func is None:
            return lambda f: st.cache(f, allow_output_mutation=True)
        return st.cache(func, allow_output_mutation=True)

# --- load SavedModel and signature ---
@cache_decorator
def load_signature(saved_model_dir="./models"):
    if not os.path.exists(saved_model_dir):
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")
    loaded = tf.saved_model.load(saved_model_dir)
    signatures = getattr(loaded, "signatures", {})
    sig_keys = list(signatures.keys())
    if "serving_default" in sig_keys:
        sig = signatures["serving_default"]
    elif sig_keys:
        sig = signatures[sig_keys[0]]
    else:
        # no signatures exposed, attempt to use __call__ if present
        if callable(loaded):
            sig = loaded.__call__
        else:
            raise ValueError("No callable signatures found in SavedModel.")
    # determine expected input name(s) if available
    structured = getattr(sig, "structured_input_signature", ((), {}))
    _, kwargs = structured
    input_names = list(kwargs.keys()) if isinstance(kwargs, dict) else []
    return sig, input_names

# --- preprocessing (float normalized) ---
def preprocess_image_bytes(image_bytes, target_size=(224, 224)):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, list(target_size))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # batch
    return img

# --- call signature robustly using the first input name if present ---
def call_signature(signature_fn, input_tensor, input_names):
    # prefer named input if signature expects names
    try:
        if input_names:
            call_kwargs = {input_names[0]: input_tensor}
            out = signature_fn(**call_kwargs)
        else:
            out = signature_fn(input_tensor)
    except Exception:
        # fallback: try passing tensor directly
        out = signature_fn(input_tensor)
    # if dict, pick first value
    if isinstance(out, dict):
        first_val = list(out.values())[0]
        return first_val.numpy()
    # if Tensor or EagerTensor
    try:
        return out.numpy()
    except Exception:
        # attempt conversion for nested structures
        if hasattr(out, "numpy"):
            return out.numpy()
        raise

# --- class labels (your mapping) ---
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]

# --- main app flow ---
saved_model_dir = st.text_input("SavedModel directory (relative path)", value="./models")

# load signature
try:
    signature_fn, input_names = load_signature(saved_model_dir)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# input URL
default_url = (
    "https://beanipm.pbgworks.org/sites/pbg-beanipm7/files/styles/picture_custom_user_wide_1x/"
    "public/AngularLeafSpotFig1a.jpg"
)
image_url = st.text_input("Image URL", value=default_url)

if image_url:
    # fetch image
    try:
        resp = requests.get(image_url, timeout=12)
        resp.raise_for_status()
        content = resp.content
    except Exception as e:
        st.error(f"Failed to fetch image: {e}")
        st.stop()

    # show preview
    try:
        pil = Image.open(BytesIO(content)).convert("RGB")
        st.image(pil, caption="Input image", use_column_width=True)
    except Exception:
        # if preview fails, continue (prediction may still work)
        pass

    # preprocess and predict
    try:
        input_tensor = preprocess_image_bytes(content, target_size=(224, 224))
        raw_out = call_signature(signature_fn, input_tensor, input_names)
        preds = np.asarray(raw_out)
        # ensure shape: (1, n) or (n,)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)
        # if outputs don't look like probabilities, apply softmax
        row_sum = float(np.sum(preds, axis=1)[0])
        if not (0.99 <= row_sum <= 1.01):
            probs = tf.nn.softmax(preds, axis=1).numpy()
        else:
            probs = preds
        probs = np.asarray(probs)
        top_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0, top_idx]) * 100.0
        label = CLASSES[top_idx] if 0 <= top_idx < len(CLASSES) else f"index_{top_idx}"

        # crisp single output
        st.markdown(f"**Predicted:** `{label}` — **confidence:** {confidence:.2f}%")

        # helpful small note if model seems suspicious (same class often)
        if confidence < 0.40:
            st.warning("Low confidence — prediction may be unreliable.")
        # if model constantly predicts same class for different images, user must retrain/resave
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

# small footer advice
st.write(" ")
st.caption("If the app repeatedly predicts the same label for different images, the issue is likely the model (training/export). Re-export a Keras `.keras` or `.h5` model including the final classification head, or retrain with balanced data.")
