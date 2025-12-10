# imgwebapp_debug.py
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

st.set_page_config(page_title="Bean Classifier (debug)", layout="centered")
st.title("Bean Image Classifier — Debug Mode")
st.markdown(
    "This debug app will attempt multiple ways to call your SavedModel and show raw outputs. "
    "Use it to identify why the model returns the same class for every image."
)

# -------------------------
# compatibility for caching
# -------------------------
if hasattr(st, "cache_resource"):
    cache_decorator = st.cache_resource
else:
    def cache_decorator(func=None, **kwargs):
        if func is None:
            return lambda f: st.cache(f, allow_output_mutation=True)
        return st.cache(func, allow_output_mutation=True)

@cache_decorator
def load_saved_model(saved_model_dir="./models"):
    if not os.path.exists(saved_model_dir):
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")
    loaded = tf.saved_model.load(saved_model_dir)
    signatures = getattr(loaded, "signatures", {})
    sig_keys = list(signatures.keys())
    return loaded, sig_keys

def prepare_image_bytes(image_bytes, target_size=(224, 224), dtype=tf.uint8):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, list(target_size))
    if dtype == tf.float32:
        img = tf.cast(img, tf.float32) / 255.0
    else:
        img = tf.cast(img, tf.uint8)
    img = tf.expand_dims(img, axis=0)
    return img

def try_call_signature(signature_fn, input_tensor, input_spec):
    """
    Try several strategies and return a dict with attempt results.
    Each attempt contains: success(bool), exception(str or None), output_array (or None), used_kwargs (dict)
    """
    attempts = {}

    # attempt 1: if signature expects kwargs, use the first kw name
    args, kwargs = input_spec if input_spec is not None else ((), {})
    # meta info for debugging
    attempts_meta = []

    def _record(name, ok, out=None, exc=None, used_kwargs=None):
        attempts[name] = {
            "success": ok,
            "exception": repr(exc) if exc is not None else None,
            "output": out.tolist() if (ok and isinstance(out, np.ndarray)) else out,
            "used_kwargs": used_kwargs
        }

    # 1: kwargs first-key
    try:
        if kwargs:
            k = list(kwargs.keys())[0]
            res = signature_fn(**{k: input_tensor})
        else:
            res = signature_fn(input_tensor)
        # normalize
        if isinstance(res, dict):
            first_val = list(res.values())[0].numpy()
        else:
            first_val = res.numpy()
        _record("first_try", True, first_val, None, {"mode": "kwargs-first" if kwargs else "positional"})
    except Exception as e:
        _record("first_try", False, None, e, {"mode": "kwargs-first" if kwargs else "positional"})

    # 2: try casting to float32 normalized
    try:
        inp_f = tf.cast(input_tensor, tf.float32) / (1.0 if input_tensor.dtype == tf.float32 else 255.0)
        if kwargs:
            k = list(kwargs.keys())[0]
            res = signature_fn(**{k: inp_f})
        else:
            res = signature_fn(inp_f)
        if isinstance(res, dict):
            first_val = list(res.values())[0].numpy()
        else:
            first_val = res.numpy()
        _record("float_try", True, first_val, None, {"mode": "float-cast"})
    except Exception as e:
        _record("float_try", False, None, e, {"mode": "float-cast"})

    # 3: broadcast input to all expected kwargs (rare)
    try:
        if kwargs:
            call_kwargs = {k: input_tensor for k in kwargs.keys()}
            res = signature_fn(**call_kwargs)
            if isinstance(res, dict):
                first_val = list(res.values())[0].numpy()
            else:
                first_val = res.numpy()
            _record("broadcast_kwargs", True, first_val, None, {"mode": "broadcast-all-kwargs"})
        else:
            _record("broadcast_kwargs", False, None, "no-kwargs", None)
    except Exception as e:
        _record("broadcast_kwargs", False, None, e, {"mode": "broadcast-all-kwargs"})

    return attempts

# -------------------------
# UI controls
# -------------------------
saved_model_dir = st.text_input("SavedModel directory (relative path)", value="./models")
with st.spinner("Loading model..."):
    try:
        loaded, sig_keys = load_saved_model(saved_model_dir)
        st.success(f"Loaded SavedModel directory: {saved_model_dir}")
        st.info(f"Available signatures: {sig_keys if sig_keys else 'NONE (inspect via tf.saved_model.load)'}")
    except Exception as e:
        st.error(f"Failed to load SavedModel: {e}")
        st.write(traceback.format_exc())
        st.stop()

# choose signature if multiple
signatures = getattr(loaded, "signatures", {})
chosen_sig = st.selectbox("Choose signature to call", options=(["serving_default"] + list(signatures.keys())) if signatures else ["serving_default"])
signature_fn = None
if signatures and chosen_sig in signatures:
    signature_fn = signatures[chosen_sig]
else:
    # if not present in signatures dict, try to access loaded.signatures[chosen_sig] may fail; attempt to access attribute
    try:
        signature_fn = signatures.get(chosen_sig, None)
    except Exception:
        signature_fn = None

st.markdown("---")
st.write("**Classes (label mapping used by app):** `0->angular_leaf_spot`, `1->bean_rust`, `2->healthy`")
classes = ["angular_leaf_spot", "bean_rust", "healthy"]

# Input URL
default_url = "https://beanipm.pbgworks.org/sites/pbg-beanipm7/files/styles/picture_custom_user_wide_1x/public/AngularLeafSpotFig1a.jpg"
path = st.text_input("Enter Image URL to classify:", default_url)
manual_preproc = st.radio("Force preprocessing to use (debug):", ("auto (try uint8, then float)", "use uint8", "use float32 normalized"))
st.markdown("Click **Classify** to run debug prediction attempts.")

if st.button("Classify") and path:
    # fetch
    try:
        resp = requests.get(path, timeout=12)
        resp.raise_for_status()
        content = resp.content
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")
        st.stop()

    # preview
    try:
        pil = Image.open(BytesIO(content)).convert("RGB")
        st.image(pil, caption="Input image", use_column_width=True)
    except Exception as e:
        st.warning(f"Preview failed: {e}")

    # prepare two tensors
    tensor_uint8 = prepare_image_bytes(content, target_size=(224,224), dtype=tf.uint8)
    tensor_float = prepare_image_bytes(content, target_size=(224,224), dtype=tf.float32)

    st.write("Prepared input tensors:")
    st.write(f"- uint8 dtype: {tensor_uint8.dtype}, shape: {tensor_uint8.shape}, min/max: {tf.reduce_min(tensor_uint8).numpy()}/{tf.reduce_max(tensor_uint8).numpy()}")
    st.write(f"- float32 dtype: {tensor_float.dtype}, shape: {tensor_float.shape}, min/max: {tf.reduce_min(tensor_float).numpy()}/{tf.reduce_max(tensor_float).numpy()}")

    # if signature_fn missing, try to get serving_default by inspection
    if signature_fn is None:
        try:
            signatures = getattr(loaded, "signatures", {})
            if "serving_default" in signatures:
                signature_fn = signatures["serving_default"]
            elif sig_keys:
                signature_fn = signatures[sig_keys[0]]
            else:
                st.error("No signature function found in SavedModel.signatures. The SavedModel might not include signatures; inspect via tf.saved_model.load(...).")
                st.stop()
        except Exception as e:
            st.error(f"Failed to obtain signature function: {e}")
            st.stop()

    st.write("Calling signature:", getattr(signature_fn, "__name__", str(signature_fn)))

    # decide which inputs to try based on radio
    attempts_results = {}
    try:
        if manual_preproc == "use uint8":
            attempts_results = try_call_signature(signature_fn, tensor_uint8, getattr(signature_fn, "structured_input_signature", ((), {})))
        elif manual_preproc == "use float32":
            attempts_results = try_call_signature(signature_fn, tensor_float, getattr(signature_fn, "structured_input_signature", ((), {})))
        else:
            # auto: try uint8 first then float path will be executed by the function itself
            attempts_results = try_call_signature(signature_fn, tensor_uint8, getattr(signature_fn, "structured_input_signature", ((), {})))
            # also attempt explicitly float-only for direct compare
            float_attempts = try_call_signature(signature_fn, tensor_float, getattr(signature_fn, "structured_input_signature", ((), {})))
            # merge float attempts under a grouped key
            attempts_results = {"uint8_path": attempts_results, "float_path": float_attempts}
    except Exception as e:
        st.error(f"Failed during signature calls: {e}")
        st.write(traceback.format_exc())
        st.stop()

    st.markdown("## Raw attempt results (first output tensor shown)")
    st.json(attempts_results)

    # Helper to examine outputs and produce a readable classification
    def analyze_preds(raw):
        """raw: numpy array (batch, classes) or shape (classes,)"""
        arr = np.array(raw)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        # stats
        stats = {
            "shape": list(arr.shape),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
        # check for suspicious constant outputs
        const_check = np.allclose(arr, arr[0], atol=1e-6)
        stats["all_rows_equal"] = bool(const_check)
        # choose label by argmax
        arg = int(np.argmax(arr, axis=1)[0])
        prob = float(np.max(tf.nn.softmax(arr, axis=1).numpy()))
        return stats, arg, prob, arr

    st.markdown("## Analysis of each successful attempt")
    # flatten attempts_results so we process every attempt
    flat = []
    if "uint8_path" in attempts_results and "float_path" in attempts_results:
        for k, v in attempts_results["uint8_path"].items():
            flat.append(("uint8_path."+k, v))
        for k, v in attempts_results["float_path"].items():
            flat.append(("float_path."+k, v))
    else:
        for k, v in attempts_results.items():
            flat.append((k, v))

    any_success = False
    for name, result in flat:
        st.markdown(f"### Attempt: `{name}`")
        st.write("Used kwargs:", result.get("used_kwargs"))
        if not result.get("success"):
            st.error(f"Attempt failed: {result.get('exception')}")
            continue
        any_success = True
        raw_out = result.get("output")
        # ensure numpy
        raw_arr = np.array(raw_out)
        st.write("Raw output (first 8 values shown):")
        # show truncated view if large
        if raw_arr.size > 50:
            st.write(raw_arr.flatten()[:50].tolist(), "... (truncated)")
        else:
            st.write(raw_arr.tolist())

        stats, arg, prob, arr = analyze_preds(raw_arr)
        st.write("Stats:", stats)
        st.write(f"Argmax index: {arg}")
        # decide mapping safety
        if stats["all_rows_equal"]:
            st.warning("Model returned identical outputs across different inputs (all rows equal). This indicates: model may be constant, or signature returns deterministic/constant vector, or you're inspecting an embedding instead of logits.")
        # Check dimensionality: if arr.shape[1] is large (e.g., > 100), likely embeddings not class logits
        if arr.ndim >= 2 and arr.shape[1] > 50:
            st.warning(f"Output dimension is large ({arr.shape[1]}) — likely an embedding/feature vector, not class logits. If so, you need to attach classification head or call correct signature.")
        # softmaxed probability of chosen class
        try:
            soft = tf.nn.softmax(arr, axis=1).numpy()
            chosen_prob = float(soft[0, arg])
            st.write(f"Softmax probability for predicted class: {chosen_prob:.4f}")
        except Exception:
            st.write("Could not compute softmax (output may be non-numeric).")
        # map to class label if possible
        if 0 <= arg < len(classes):
            st.success(f"Predicted label (mapping): {arg} -> {classes[arg]}")
        else:
            st.warning(f"Predicted index {arg} outside class mapping (0..{len(classes)-1}).")

    if not any_success:
        st.error("All attempts to call the SavedModel signature failed. Inspect model signatures or re-export the model as a Keras `.keras` or `.h5` file for easier loading.")
    else:
        st.info("Use the above analysis to determine cause. See suggestions below.")

    st.markdown("---")
    st.markdown("## Suggestions (diagnosis checklist)")
    st.markdown("""
    1. **Preprocessing mismatch** — confirm the model's training preprocessing (input size, scaling, mean subtraction). Try forcing `use float32` or `use uint8` above and observe which gives sensible probabilities.
    2. **Wrong signature / outputs embedding** — if the output dimensionality is large (>50) it's likely an embedding. Check your SavedModel signatures; you may have exported a feature extractor rather than the final classification model. Re-export the full model with the classification head.
    3. **Model collapsed to constant** — if outputs are identical across different images, your model weights could be constant or corrupted. Re-load the original training checkpoint and re-evaluate on holdout images.
    4. **Check SavedModel signatures in a separate Python REPL**:
       ```py
       import tensorflow as tf
       loaded = tf.saved_model.load("./models")
       print(getattr(loaded, "signatures", {}).keys())
       fn = loaded.signatures.get("serving_default")
       print(fn.structured_input_signature)
       print(fn.structured_outputs)
       ```
    5. **If you can re-export**: train/export as a Keras `.keras` file (supports `tf.keras.models.load_model()` with Keras 3):
       ```py
       model.save("bean_model.keras")
       ```
    6. **If signature returns dict with multiple outputs**, ensure you pick the correct output key (e.g., `"predictions"` or `"logits"`).
    """)

