cd ~/Desktop/emoji-qh

# Tạo file app.py mới
cat > app.py << 'EOF'
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="Emoji Classifier", page_icon="🎨", layout="centered")

st.title("🎨 Nhận diện Emoji với AI")
st.markdown("*Vẽ hoặc upload ảnh emoji - Hệ thống sẽ nhận diện*")

@st.cache_resource
def load_model():
    model_path = 'emoji_modelqh5.h5'
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy file model: {model_path}")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

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
    img_array = img_array.reshape(1, 28, 28)
    return img_array

model = load_model()
class_names = get_class_names()

if model is None:
    st.stop()

tab1, tab2 = st.tabs(["✏️ Vẽ Emoji", "📤 Upload ảnh"])

with tab1:
    drawing = st.canvas(
        "",
        width=280,
        height=280,
        background_color="#FFFFFF",
        stroke_color="#000000",
        stroke_width=15,
        update_streamlit=True,
        key="canvas"
    )
    
    if drawing.image_data is not None:
        if st.button("🔍 Nhận diện", key="predict_draw"):
            with st.spinner("Đang xử lý..."):
                canvas_img = Image.fromarray(drawing.image_data.astype('uint8'), mode='RGBA')
                canvas_img = canvas_img.convert('L')
                processed = preprocess_image(canvas_img)
                predictions = model.predict(processed, verbose=0)[0]
                predicted_idx = np.argmax(predictions)
                predicted_label = class_names[predicted_idx]
                confidence = predictions[predicted_idx]
                st.success(f"### 🎯 Kết quả: {predicted_label}")
                st.metric("Độ tin cậy", f"{confidence:.2%}")

with tab2:
    uploaded_file = st.file_uploader("Chọn ảnh emoji", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh của bạn", width=200)
        
        if st.button("🔍 Nhận diện", key="predict_upload"):
            with st.spinner("Đang xử lý..."):
                processed = preprocess_image(image)
                predictions = model.predict(processed, verbose=0)[0]
                predicted_idx = np.argmax(predictions)
                predicted_label = class_names[predicted_idx]
                confidence = predictions[predicted_idx]
                st.success(f"### 🎯 Kết quả: {predicted_label}")
                st.metric("Độ tin cậy", f"{confidence:.2%}")

def get_emoji_icon(label):
    icons = {
        'cloud': '☁️',
        'grinning_face': '😀',
        'heart': '❤️',
        'smiling_horns': '👿',
        'thumb': '👍'
    }
    return icons.get(label, '🎨')

def format_label(label):
    names = {
        'cloud': 'Cloud',
        'grinning_face': 'Smiley Face',
        'heart': 'Heart',
        'smiling_horns': 'Horned Smiley',
        'thumb': 'Thumbs Up'
    }
    return names.get(label, label)

st.markdown("---")
st.caption("Model được train trên dataset emoji với MLP architecture")
EOF

# Commit và push
git add app.py
git commit -m "Update app.py with better UI"
git push origin main
