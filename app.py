import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Skin Lesion Segmentation", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('unet_skin_lesion_model.h5')

model = load_model()

st.title("Skin Lesion Boundary Detector")
st.write("**Body part:** Skin | **Type:** Dermoscopic photo | **Task:** Outlines the boundary of a skin lesion")
st.warning("This is a student coursework prototype, not a medical device. It is not intended for real diagnostic use.")

uploaded_file = st.file_uploader("Upload a dermoscopic image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img_resized = image.resize((128, 128))
    img_arr = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_arr, axis=0)

    pred_mask = model.predict(img_input)[0]
    pred_mask_binary = (pred_mask.squeeze() > 0.5).astype(np.uint8)

    lesion_pixel_pct = (pred_mask_binary.sum() / pred_mask_binary.size) * 100

    overlay = np.array(img_resized).copy()
    overlay[pred_mask_binary == 1] = [255, 0, 0]
    blended = (0.6 * np.array(img_resized) + 0.4 * overlay).astype(np.uint8)

    st.image(blended, caption='Predicted Lesion Boundary (red overlay)', use_container_width=True)

    if lesion_pixel_pct < 0.5:
        st.info("The model didn't detect a clear lesion boundary in this image. This may mean no lesion is visible, or the image differs too much from the training data (e.g. lighting, zoom, or image type) for a reliable prediction.")
    else:
        st.success(f"The model predicts a lesion covering approximately {lesion_pixel_pct:.1f}% of the image area, highlighted in red above.")