import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="Emoji Classifier", page_icon="🎨")
st.title("🎨 Nhận diện Emoji với AI")

# DEBUG: Xem thư mục có file gì
st.write("### Debug: Các file trong thư mục hiện tại:")
st.write(os.listdir('.'))

# Tìm file model
h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
keras_files = [f for f in os.listdir('.') if f.endswith('.keras')]

st.write(f"File .h5 tìm thấy: {h5_files}")
st.write(f"File .keras tìm thấy: {keras_files}")

@st.cache_resource
def load_model():
    # Thử load .keras trước
    if 'emoji_modelqh5.keras' in keras_files:
        st.success("✅ Đang load file .keras")
        return tf.keras.models.load_model('emoji_modelqh5.keras')
    # Thử load .h5
    elif 'emoji_modelqh5.h5' in h5_files:
        st.success("✅ Đang load file .h5")
        return tf.keras.models.load_model('emoji_modelqh5.h5')
    else:
        st.error("❌ Không tìm thấy file model nào!")
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
            st.success(f"### 🎯 Kết quả: {label}")

with tab2:
    uploaded = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'])
    if uploaded and st.button("🔍 Nhận diện", key="upload"):
        img = Image.open(uploaded)
        st.image(img, width=200)
        processed = preprocess_image(img)
        pred = model.predict(processed, verbose=0)[0]
        label = class_names[np.argmax(pred)]
        st.success(f"### 🎯 Kết quả: {label}")
