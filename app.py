import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="Emoji Classifier", page_icon="🎨")
st.title("🎨 Nhận diện Emoji với AI")

@st.cache_resource
def load_model():
    try:
        # Load file .keras
        model = tf.keras.models.load_model('emoji_modelqh5.keras', compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def get_class_names():
    return ['cloud', 'grinning_face', 'heart', 'smiling_horns', 'thumb']

def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    if img_array.mean() > 0.7:
        img_array = 1.0 - img_array
    target_mean = 0.242
    if img_array.mean() > 0:
        img_array = img_array * (target_mean / img_array.mean())
        img_array = np.clip(img_array, 0, 1)
    return img_array.reshape(1, 28, 28)

model = load_model()
class_names = get_class_names()

if model is None:
    st.stop()

tab1, tab2 = st.tabs(["✏️ Vẽ Emoji", "📤 Upload ảnh"])

with tab1:
    drawing = st.canvas("", width=280, height=280, background_color="#FFFFFF", 
                        stroke_color="#000000", stroke_width=15, key="canvas")
    if drawing.image_data is not None and st.button("🔍 Nhận diện", key="draw"):
        with st.spinner("Đang xử lý..."):
            img = Image.fromarray(drawing.image_data.astype('uint8'), mode='RGBA').convert('L')
            processed = preprocess_image(img)
            pred = model.predict(processed, verbose=0)[0]
            label = class_names[np.argmax(pred)]
            confidence = max(pred)
            st.success(f"### 🎯 Kết quả: {label}")
            st.metric("Độ tin cậy", f"{confidence:.2%}")

with tab2:
    uploaded = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'])
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Ảnh của bạn", width=200)
        if st.button("🔍 Nhận diện", key="upload"):
            with st.spinner("Đang xử lý..."):
                processed = preprocess_image(img)
                pred = model.predict(processed, verbose=0)[0]
                label = class_names[np.argmax(pred)]
                confidence = max(pred)
                st.success(f"### 🎯 Kết quả: {label}")
                st.metric("Độ tin cậy", f"{confidence:.2%}")

st.markdown("---")
st.caption("Emoji Classifier - MLP Model")
