import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.title("Bean Image Classifier")
st.text("Provide URL of bean image for image classification")

# Use cache_resource if available (Streamlit >= ~1.18); otherwise fall back to st.cache
if hasattr(st, "cache_resource"):
    cache_decorator = st.cache_resource
else:
    # older Streamlit versions
    def cache_decorator(func=None, **kwargs):
        if func is None:
            return lambda f: st.cache(f, allow_output_mutation=True)
        return st.cache(func, allow_output_mutation=True)

@cache_decorator
def load_model(model_path="/app/models/"):
    """Load and return the Keras model. Adjust model_path if different."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # re-raise so caller can handle and display a friendly message
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

with st.spinner("Loading model into memory..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

classes = ["angular_leaf_spot", "bean_rust", "healthy"]

def decode_img(image_bytes):
    """
    Decode JPEG/PNG bytes into a batched float32 tensor,
    resized to 224x224 and normalized to [0, 1].
    """
    # convert to tf tensor of type string
    img_tensor = tf.io.decode_jpeg(image_bytes, channels=3)  # works for JPEG; PNG usually OK too
    img_tensor = tf.image.resize(img_tensor, [224, 224])
    img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # batch dim
    return img_tensor  # tf.Tensor with shape (1, 224, 224, 3)

# Default example image URL (you can change)
default_url = "https://beanipm.pbgworks.org/sites/pbg-beanipm7/files/styles/picture_custom_user_wide_1x/public/AngularLeafSpotFig1a.jpg"
path = st.text_input("Enter Image URL to classify:", default_url)

if path:
    # fetch image bytes
    try:
        resp = requests.get(path, timeout=10)
        resp.raise_for_status()
        content = resp.content
    except requests.RequestException as e:
        st.error(f"Failed to fetch image from URL: {e}")
    else:
        # show image preview
        try:
            image = Image.open(BytesIO(content)).convert("RGB")
            st.image(image, caption="Input image", use_column_width=True)
        except Exception as e:
            st.warning(f"Could not render preview image (but bytes were fetched): {e}")

        # run prediction
        with st.spinner("Classifying..."):
            try:
                input_tensor = decode_img(content)               # tf.Tensor
                preds = model.predict(input_tensor)              # returns numpy array or tensor
                label_idx = int(np.argmax(preds, axis=1)[0])     # get index
                st.write("Predicted class:")
                st.success(classes[label_idx])
                # optional: print probabilities
                prob_list = preds[0].tolist()
                for i, c in enumerate(classes):
                    st.write(f"{c}: {prob_list[i]:.4f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
