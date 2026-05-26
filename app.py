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

st.set_page_config(page_title="Emoji Classifier", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Source Code Pro', 'Courier New', monospace !important;
    }
    
    .big-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }
    
    .sub-text {
        font-size: 0.8rem;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    
    /* Ẩn mấy thứ linh tinh */
    #MainMenu, footer, header, .stActionButton, .stCanvasToolbar {
        display: none !important;
    }
    
    /* Căn giữa TOÀN BỘ nội dung */
    .main .block-container {
        max-width: 500px !important;
        padding-top: 2rem !important;
        margin: 0 auto !important;
    }
    
    /* Canvas căn giữa */
    .canvas-wrapper {
        display: flex !important;
        justify-content: center !important;
        margin-bottom: 1rem !important;
    }
    
    /* Nút căn giữa */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        margin-top: 1rem !important;
    }
    
    /* NÚT 3D - TRẮNG VIỀN ĐEN */
    .stButton button {
        background: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 40px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        box-shadow: 0 6px 0 #000000 !important;
        transition: none !important;
    }
    
    /* KHÔNG HOVER */
    .stButton button:hover {
        background: #FFFFFF !important;
        transform: none !important;
        box-shadow: 0 6px 0 #000000 !important;
    }
    
    /* NHẤN LÚN */
    .stButton button:active {
        transform: translateY(3px) !important;
        box-shadow: 0 3px 0 #000000 !important;
    }
    
    .prediction-box {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 1rem;
        margin-top: 1.5rem;
        border-bottom: 2px solid #000000;
    }
    
    .confidence-text {
        text-align: center;
        font-size: 0.75rem;
        color: #5f6368;
        margin-top: 0.5rem;
    }
    
    .prob-text {
        text-align: center;
        font-size: 0.7rem;
        margin-top: 1rem;
        line-height: 1.6;
    }
    
    hr {
        margin-top: 2rem;
        border-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">EMOJI CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">by MLP model v1.0 - 2026</div>', unsafe_allow_html=True)

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
    
    with st.spinner("training model..."):
        model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=0)
    
    return model, unique_labels

def preprocess_image(image):
    img = image.convert('L').resize((28, 28))
    img_array = 1.0 - (np.array(img).astype('float32') / 255.0)
    return img_array.reshape(1, 28, 28)

model, class_names = load_and_train_model()

class_display = {
    'cloud': 'CLOUD',
    'grinning_face': 'SMILEY',
    'heart': 'HEART',
    'smiling_horns': 'HORNED',
    'thumb': 'THUMB'
}

# ------------------- GIAO DIỆN -------------------
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.last_probs = None

# KHÔNG dùng columns nữa - căn giữa tự nhiên
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

if st.button("CONFIRM!"):
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:, :, 3]) > 100:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
        pred = model.predict(preprocess_image(img), verbose=0)[0]
        st.session_state.last_probs = pred
        st.session_state.prediction = class_names[np.argmax(pred)]
        st.session_state.confidence = max(pred)
    else:
        st.warning("draw something first")

if st.session_state.prediction:
    st.markdown(f"""
    <div class="prediction-box">{class_display.get(st.session_state.prediction, st.session_state.prediction.upper())}</div>
    <div class="confidence-text">confidence: {st.session_state.confidence:.2%}</div>
    """, unsafe_allow_html=True)
    
    if st.session_state.last_probs is not None:
        prob_text = "  |  ".join([f"{n}: {st.session_state.last_probs[i]:.2%}" for i, n in enumerate(class_names)])
        st.markdown(f'<div class="prob-text">{prob_text}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align: center; font-size: 0.7rem; color: #9aa0a6;">Source Code Pro · TensorFlow · MLP</div>', unsafe_allow_html=True)
