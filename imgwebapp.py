# imgwebapp.py
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import traceback
from io import BytesIO

import numpy as np
import requests
from PIL import Image

import tensorflow as tf
import streamlit as st

st.set_page_config(page_title="Bean Image Classifier", layout="centered")
st.title("Bean Image Classifier (SavedModel loader)")
st.text("Provide URL of bean image or use the default example to classify.")

# -------------------------
# caching decorator (Streamlit compatibility)
# -------------------------
if hasattr(st, "cache_resource"):
    cache_decorator = st.cache_resource
else:
    # fallback for older Streamlit
    def cache_decorator(func=None, **kwargs):
        if func is None:
            return lambda f: st.cache(f, allow_output_mutation=True)
        return st.cache(func, allow_output_mutation=True)

# -------------------------
# load saved model and signature
# -------------------------
@cache_decorator
def load_saved_model(saved_model_dir="./models"):
    """
    Load a TensorFlow SavedModel from directory and select a callable signature.
    Returns: (loaded, signature_name, signature_fn, signature_input_specs)
    """
    if not os.path.exists(saved_model_dir):
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")

    loaded = tf.saved_model.load(saved_model_dir)
    # available signatures (dict-like). Many SavedModels have 'serving_default'
    sig_keys = list(getattr(loaded, "signatures", {}).keys())
    if not sig_keys:
        # loaded may still have concrete functions accessible via loaded.signatures missing;
        # try listing attributes or use `loaded.__call__` if present
        raise ValueError(
            f"No callable signatures found in SavedModel at {saved_model_dir}. "
            "Inspect the SavedModel with `tf.saved_model.load(...).signatures`."
        )

    # Prefer serving_default if available
    chosen = "serving_default" if "serving_default" in sig_keys else sig_keys[0]
    signature_fn = loaded.signatures[chosen]

    # get structured input signature so we can know expected input names/specs
    structured = getattr(signature_fn, "structured_input_signature", None)
    # structured_input_signature returns (args, kwargs) where kwargs are named inputs
    argspec = structured if structured is not None else ((), {})
    return loaded, chosen, signature_fn, argspec

# -------------------------
# helper to prepare image tensor
# -------------------------
def prepare_image_bytes(image_bytes, target_size=(224, 224), dtype=tf.uint8):
    """
    Decode image bytes and return a batched tensor of dtype (tf.uint8 or tf.float32).
    - target_size: (h, w)
    - dtype: desired dtype of returned tensor (tf.uint8 or tf.float32)
    """
    # decode - handles JPEG/PNG auto-detection
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, list(target_size))
    if dtype == tf.float32:
        img = tf.cast(img, tf.float32) / 255.0
    else:
        img = tf.cast(img, tf.uint8)
    img = tf.expand_dims(img, axis=0)  # add batch dim
    return img

# -------------------------
# call saved model signature with several fallbacks
# -------------------------
def predict_with_signature(signature_fn, input_tensor, input_spec):
    """
    Try a few common ways to call the signature_fn. Returns numpy array of predictions.
    - signature_fn: concrete function (callable)
    - input_tensor: batched tf.Tensor (1, H, W, C)
    - input_spec: structured_input_signature from signature_fn (args, kwargs)
    """
    # 1) If signature expects named kwargs, build dict
    args, kwargs = input_spec if input_spec is not None else ((), {})
    # kwargs is a dict mapping name -> TensorSpec (often)
    try:
        if kwargs:
            # pick the first kw name expected
            input_name = list(kwargs.keys())[0]
            call_kwargs = {input_name: input_tensor}
            out = signature_fn(**call_kwargs)
        else:
            # try calling with positional tensor
            out = signature_fn(input_tensor)
    except Exception as e:
        # try alternate attempt: if signature expects float inputs, cast and try
        try:
            inp_float = tf.cast(input_tensor, tf.float32) / (1.0 if input_tensor.dtype == tf.float32 else 255.0)
            if kwargs:
                input_name = list(kwargs.keys())[0]
                call_kwargs = {input_name: inp_float}
                out = signature_fn(**call_kwargs)
            else:
                out = signature_fn(inp_float)
        except Exception as e2:
            # last resort: try passing a dict with all kwargs as same tensor (rare)
            try:
                if kwargs:
                    call_kwargs = {k: input_tensor for k in kwargs.keys()}
                    out = signature_fn(**call_kwargs)
                else:
                    raise e2
            except Exception as e3:
                # raise a combined informative error
                raise RuntimeError(
                    "Failed to call SavedModel signature using several strategies. "
                    f"Errors:\n - first: {e}\n - second (float cast): {e2}\n - third (broadcast to all kwargs): {e3}\n"
                )

    # signature_fn returns a mapping (dict) of outputs or a tensor. Normalize to numpy array
    if isinstance(out, dict):
        # pick the first tensor in the dict
        first_value = list(out.values())[0]
        preds = first_value.numpy()
    else:
        preds = out.numpy()
    return preds

# -------------------------
# Streamlit UI + inference
# -------------------------
default_url = (
    "https://beanipm.pbgworks.org/sites/pbg-beanipm7/files/styles/picture_custom_user_wide_1x/"
    "public/AngularLeafSpotFig1a.jpg"
)

saved_model_dir = st.text_input("SavedModel directory (relative path)", value="./models")

# load model / signature once
with st.spinner("Loading SavedModel..."):
    try:
        loaded, signature_name, signature_fn, signature_input = load_saved_model(saved_model_dir)
        st.success(f"Loaded SavedModel from `{saved_model_dir}` — signature: `{signature_name}`")
    except Exception as e:
        st.error(f"Failed to load SavedModel: {e}")
        st.write(traceback.format_exc())
        st.stop()

classes = ["angular_leaf_spot", "bean_rust", "healthy"]

path = st.text_input("Enter Image URL to classify:", default_url)

if path:
    try:
        resp = requests.get(path, timeout=12)
        resp.raise_for_status()
        content = resp.content
    except Exception as e:
        st.error(f"Failed to fetch image: {e}")
        st.stop()

    # show preview
    try:
        pil = Image.open(BytesIO(content)).convert("RGB")
        st.image(pil, caption="Input image", use_column_width=True)
    except Exception as e:
        st.warning(f"Could not render preview: {e}")

    # prepare tensor and perform prediction
    with st.spinner("Classifying..."):
        try:
            # attempt uint8 first (many TF SavedModels accept uint8 images)
            input_tensor_uint8 = prepare_image_bytes(content, target_size=(224, 224), dtype=tf.uint8)
            try:
                preds = predict_with_signature(signature_fn, input_tensor_uint8, signature_input)
            except RuntimeError:
                # try float32 normalized
                input_tensor_float = prepare_image_bytes(content, target_size=(224, 224), dtype=tf.float32)
                preds = predict_with_signature(signature_fn, input_tensor_float, signature_input)

            # preds should be NumPy array of shape (1, num_classes) or similar
            preds = np.asarray(preds)
            if preds.ndim == 1:
                # shape (num_classes,) -> expand
                preds = np.expand_dims(preds, axis=0)

            label_idx = int(np.argmax(preds, axis=1)[0])
            st.write("Predicted class:")
            if 0 <= label_idx < len(classes):
                st.success(classes[label_idx])
            else:
                st.warning(f"Predicted index {label_idx} outside class list length ({len(classes)}).")
                st.write("Raw predictions:", preds.tolist())

            # print probabilities (if available)
            st.write("Probabilities / model outputs:")
            for i in range(min(preds.shape[1], len(classes))):
                st.write(f"{classes[i]}: {preds[0, i]:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write(traceback.format_exc())
            st.stop()
