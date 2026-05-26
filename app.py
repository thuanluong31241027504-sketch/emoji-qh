import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas
import os

# ------------------- CẤU HÌNH TRANG -------------------
st.set_page_config(page_title="Let's Draw - Emoji", layout="centered")

# CSS Minimal: Chỉ giữ lại nét bút và bố cục cơ bản
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playpen+Sans:wght@400;500;600;700&display=swap');
    
    html, body, .stApp, .main, div, p, span, h1, h2, h3, h4, button, label {
        font-family: 'Playpen Sans', 'Comic Neue', cursive !important;
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Header chính */
    .big-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0rem;
        color: #000000;
    }
    .sub-text {
        font-size: 1rem;
        font-weight: 400;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    
    /* Ẩn toàn bộ phần tử thừa của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stAlert {background-color: #f8f9fa; border-left: 3px solid #000;}
    
    /* Nút bấm Minimal */
    .stButton button {
        background-color: #000000;
        color: #FFFFFF;
        border: 1px solid #000000;
        border-radius: 0px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: 0.2s;
    }
    .stButton button:hover {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #000000;
        box-shadow: none;
    }
    
    /* Kết quả dự đoán */
    .prediction-box {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 1rem;
        margin-top: 1rem;
        border-top: 2px solid #e0e0e0;
        border-bottom: 2px solid #e0e0e0;
    }
    .confidence-text {
        text-align: center;
        font-size: 0.9rem;
        color: #5f6368;
        margin-top: 0.5rem;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header Minimal
st.markdown('<div class="big-title">✏️ Let’s Draw.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Draw a simple emoji. AI will try to guess.</div>', unsafe_allow_html=True)

# ------------------- HÀM TRAIN MODEL (GIỮ NGUYÊN) -------------------
@st.cache_resource
def load_and_train_model():
    with st.spinner("Training AI model... (First run will take ~2-3 min)"):
        if not os.path.exists("emoji-dataset"):
            os.system('git clone https://github.com/thuanluong31241027504-sketch/emoji-dataset.git')
        
        data = []
        labels = []
        emoji_path = 'emoji-dataset/my_emoji_dataset'
        
        for emoji_name in os.listdir(emoji_path):
            emoji_folder = os.path.join(emoji_path, emoji_name)
            if os.path.isdir(emoji_folder):
                for img_file in os.listdir(emoji_folder):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emoji_folder, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (28, 28))
                            data.append(img)
                            labels.append(emoji_name)
        
        X = np.array(data, dtype=np.uint8)
        y = np.array(labels)
        
        X = X.astype('float32') / 255.0
        unique_labels = np.unique(y)
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        y_numeric = np.array([label_to_id[label] for label in labels])
        y_categorical = to_categorical(y_numeric)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
        
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(unique_labels), activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=0)
        
        return model, unique_labels

def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    # Đảo màu nếu cần (Nền trắng -> Chữ đen)
    if img_array.mean() > 0.7:
        img_array = 1.0 - img_array
    target_mean = 0.242
    if img_array.mean() > 0:
        img_array = img_array * (target_mean / img_array.mean())
        img_array = np.clip(img_array, 0, 1)
    return img_array.reshape(1, 28, 28)

model, class_names = load_and_train_model()

# ------------------- GIAO DIỆN VẼ TỐI GIẢN -------------------
# Chỉ giữ lại nét vẽ đen, độ dày cố định
stroke_width = 12
stroke_color = "#000000"

# Khởi tạo session state để lưu canvas và kết quả
if "canvas_data" not in st.session_state:
    st.session_state.canvas_data = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Layout khung vẽ
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div style="text-align: center; margin-bottom: 10px;">⬇️ draw here ⬇️</div>', unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode="freedraw",  # Chỉ cho vẽ tay
        key="canvas",
    )
    
    # Lưu dữ liệu canvas vào session state
    if canvas_result.image_data is not None:
        st.session_state.canvas_data = canvas_result.image_data
    
    # Hai nút: CLEAR và CONFIRM
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🗑️ CLEAR", use_container_width=True):
            st.session_state.canvas_data = None
            st.session_state.prediction_result = None
            st.rerun()
    with btn_col2:
        if st.button("✅ CONFIRM", use_container_width=True):
            if st.session_state.canvas_data is not None:
                img = Image.fromarray(st.session_state.canvas_data.astype('uint8'), mode='RGBA')
                img_gray = img.convert('L')
                
                # Kiểm tra xem có vẽ gì không
                if np.mean(np.array(img_gray)) < 250:
                    processed_img = preprocess_image(img)
                    predictions = model.predict(processed_img, verbose=0)[0]
                    predicted_idx = np.argmax(predictions)
                    st.session_state.prediction_result = {
                        "label": class_names[predicted_idx],
                        "confidence": predictions[predicted_idx]
                    }
                else:
                    st.session_state.prediction_result = {"error": "Empty canvas"}
            else:
                st.session_state.prediction_result = {"error": "No drawing"}

# ------------------- HIỂN THỊ KẾT QUẢ -------------------
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    if "error" in result:
        st.warning("Please draw something before confirming.")
    else:
        # Chuyển đổi label để hiển thị đẹp mắt (không dùng emoji)
        display_names = {
            'cloud': 'CLOUD',
            'grinning_face': 'SMILEY',
            'heart': 'HEART',
            'smiling_horns': 'HORNED',
            'thumb': 'THUMB'
        }
        label_display = display_names.get(result['label'], result['label'].upper())
        conf = result['confidence']
        
        st.markdown("---")
        st.markdown(f'<div class="prediction-box">🤖 {label_display}</div>', unsafe_allow_html=True)
        
        if conf > 0.6:
            st.markdown(f'<div class="confidence-text">confidence: {conf:.2%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="confidence-text" style="color:#c5221f;">Try drawing clearer.</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; font-size: 0.75rem; color: #9aa0a6;">Playpen Sans · MLP Model · TensorFlow</div>', unsafe_allow_html=True)
