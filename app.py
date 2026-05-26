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
st.set_page_config(page_title="Emoji Classifier", page_icon="✏️", layout="centered")

# CSS font máy đánh chữ (monospace)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@300;400;500;600;700&display=swap');
    
    html, body, .stApp, div, p, span, h1, h2, h3, h4, button, label {
        font-family: 'Source Code Pro', 'Courier New', monospace !important;
        background-color: #FFFFFF;
        color: #000000;
    }
    
    .big-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0rem;
        color: #000000;
    }
    
    .sub-text {
        font-size: 0.9rem;
        font-weight: 400;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton button {
        background-color: #000000;
        color: #FFFFFF;
        border: 1px solid #000000;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.9rem;
        width: 100%;
        transition: 0.2s;
    }
    
    .stButton button:hover {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    .prediction-box {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 1rem;
        margin-top: 1rem;
        border: 2px solid #000000;
        background-color: #f8f9fa;
    }
    
    .confidence-text {
        text-align: center;
        font-size: 0.8rem;
        color: #5f6368;
        margin-top: 0.5rem;
    }
    
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #e0e0e0;
    }
    
    .stAlert {
        background-color: #f8f9fa;
        border-left: 3px solid #000000;
        font-family: 'Source Code Pro', monospace;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">✏️ EMOJI CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">draw → confirm → AI predicts</div>', unsafe_allow_html=True)

# ------------------- HÀM TRAIN MODEL -------------------
@st.cache_resource
def load_and_train_model():
    if not os.path.exists("emoji-dataset"):
        with st.spinner("loading dataset..."):
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
    
    with st.spinner("training model... (first time ~2-3 min)"):
        model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=0)
    
    return model, unique_labels

def preprocess_image(image):
    """Tiền xử lý ảnh từ canvas"""
    # Chuyển sang grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize về 28x28
    image = image.resize((28, 28))
    
    # Chuyển thành numpy array và chuẩn hóa
    img_array = np.array(image).astype('float32') / 255.0
    
    # Đảo màu: ảnh vẽ có nền trắng (1), nét đen (0)
    # Model train với nền đen (0), nét trắng (1) -> cần đảo
    img_array = 1.0 - img_array
    
    # Reshape cho model
    img_array = img_array.reshape(1, 28, 28)
    
    return img_array

# Load model
model, class_names = load_and_train_model()

# Class names để hiển thị
class_display = {
    'cloud': 'CLOUD',
    'grinning_face': 'SMILEY',
    'heart': 'HEART',
    'smiling_horns': 'HORNED',
    'thumb': 'THUMB'
}

# ------------------- GIAO DIỆN -------------------
# Khởi tạo session state
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# Canvas vẽ
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div style="text-align: center; margin-bottom: 10px;">┌─────────────────┐</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 10px;">│   DRAW HERE     │</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 10px;">└─────────────────┘</div>', unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=12,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    # Hai nút
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("CLEAR"):
            st.session_state.canvas_key += 1
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.rerun()
    
    with btn_col2:
        if st.button("CONFIRM"):
            if canvas_result.image_data is not None:
                # Kiểm tra xem có vẽ gì không
                img_array = canvas_result.image_data
                if np.sum(img_array[:, :, 3]) > 100:  # Có nét vẽ
                    img = Image.fromarray(img_array.astype('uint8'), mode='RGBA')
                    processed = preprocess_image(img)
                    predictions = model.predict(processed, verbose=0)[0]
                    predicted_idx = np.argmax(predictions)
                    st.session_state.prediction = class_names[predicted_idx]
                    st.session_state.confidence = predictions[predicted_idx]
                else:
                    st.session_state.prediction = None
                    st.session_state.confidence = None
                    st.warning("Please draw something first")
            else:
                st.warning("Please draw something first")

# Hiển thị kết quả
if st.session_state.prediction:
    st.markdown("---")
    st.markdown(f"""
    <div class="prediction-box">
        {class_display.get(st.session_state.prediction, st.session_state.prediction.upper())}
    </div>
    <div class="confidence-text">
        confidence: {st.session_state.confidence:.2%}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; font-size: 0.7rem; color: #9aa0a6;">Source Code Pro · 5 classes · TensorFlow</div>', unsafe_allow_html=True)
