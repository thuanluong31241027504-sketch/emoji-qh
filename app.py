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

# CSS gọn - chỉ căn giữa
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Source Code Pro', 'Courier New', monospace !important;
    }
    
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }
    
    .sub {
        font-size: 0.8rem;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    
    /* Ẩn mấy thứ linh tinh */
    #MainMenu, footer, header {
        display: none !important;
    }
    
    /* Căn giữa canvas */
    .stCanvas {
        display: flex !important;
        justify-content: center !important;
    }
    
    /* Căn giữa nút */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        margin-top: 1.5rem !important;
    }
    
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
    
    .stButton button:hover {
        background: #FFFFFF !important;
        transform: none !important;
        box-shadow: 0 6px 0 #000000 !important;
    }
    
    .stButton button:active {
        transform: translateY(3px) !important;
        box-shadow: 0 3px 0 #000000 !important;
    }
    
    .prediction {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 1rem;
        margin-top: 1.5rem;
        border-bottom: 2px solid #000;
    }
    
    .confidence {
        text-align: center;
        font-size: 0.75rem;
        color: #5f6368;
        margin-top: 0.5rem;
    }
    
    .probs {
        text-align: center;
        font-size: 0.7rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">EMOJI CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">by MLP model v1.0 - 2026</div>', unsafe_allow_html=True)

# ------------------- MODEL -------------------
@st.cache_resource
def load_model():
    if not os.path.exists("emoji-dataset"):
        os.system('git clone https://github.com/thuanluong31241027504-sketch/emoji-dataset.git')
    
    data, labels = [], []
    path = 'emoji-dataset/my_emoji_dataset'
    
    for name in os.listdir(path):
        folder = os.path.join(path, name)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        data.append(cv2.resize(img, (28, 28)))
                        labels.append(name)
    
    X = np.array(data, dtype=np.uint8).astype('float32') / 255.0
    unique = np.unique(labels)
    y = to_categorical(np.array([np.where(unique == l)[0][0] for l in labels]))
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(unique), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=0)
    
    return model, unique

model, classes = load_model()

display = {
    'cloud': 'CLOUD',
    'grinning_face': 'SMILEY',
    'heart': 'HEART',
    'smiling_horns': 'HORNED',
    'thumb': 'THUMB'
}

# ------------------- GIAO DIỆN -------------------
if 'key' not in st.session_state:
    st.session_state.key = 0
    st.session_state.pred = None
    st.session_state.conf = None
    st.session_state.probs = None

# Canvas
canvas = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.key}",
)

# Nút
if st.button("CONFIRM!"):
    if canvas.image_data is not None and np.sum(canvas.image_data[:, :, 3]) > 100:
        img = Image.fromarray(canvas.image_data.astype('uint8'), mode='RGBA').convert('L').resize((28, 28))
        arr = 1.0 - np.array(img).astype('float32') / 255.0
        pred = model.predict(arr.reshape(1, 28, 28), verbose=0)[0]
        st.session_state.probs = pred
        st.session_state.pred = classes[np.argmax(pred)]
        st.session_state.conf = max(pred)
    else:
        st.warning("draw something")

# Kết quả
if st.session_state.pred:
    st.markdown(f'<div class="prediction">{display.get(st.session_state.pred, st.session_state.pred.upper())}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">{st.session_state.conf:.2%}</div>', unsafe_allow_html=True)
    
    if st.session_state.probs is not None:
        text = "  |  ".join([f"{c}: {st.session_state.probs[i]:.2%}" for i, c in enumerate(classes)])
        st.markdown(f'<div class="probs">{text}</div>', unsafe_allow_html=True)
