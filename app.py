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

# CSS - tất cả font code, nút block 3D
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600;700;800&display=swap');
    
    /* Tất cả font chữ đều là Source Code Pro */
    * {
        font-family: 'Source Code Pro', 'Courier New', monospace !important;
    }
    
    html, body, .stApp, div, p, span, h1, h2, h3, h4, h5, h6, button, label, input {
        font-family: 'Source Code Pro', 'Courier New', monospace !important;
    }
    
    .big-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0rem;
        letter-spacing: -1px;
    }
    
    .sub-text {
        font-size: 0.8rem;
        font-weight: 400;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    
    /* Ẩn mấy thứ linh tinh */
    #MainMenu, footer, header, .stActionButton, .stCanvasToolbar {
        display: none !important;
    }
    
    /* Container nút - căn giữa */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        margin-top: 2rem !important;
    }
    
    /* Nút hình vuông, khối Block 3D */
    .stButton button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        font-family: 'Source Code Pro', monospace !important;
        width: auto !important;
        min-width: 160px !important;
        cursor: pointer !important;
        letter-spacing: 1.5px !important;
        
        /* Hiệu ứng Block 3D */
        box-shadow: 0 6px 0 #000000 !important;
        transition: all 0.05s linear !important;
        
        /* Xóa viền focus */
        outline: none !important;
    }
    
    /* Hover - giữ nguyên, không đổi */
    .stButton button:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        box-shadow: 0 6px 0 #000000 !important;
        transform: none !important;
        outline: none !important;
    }
    
    /* Hiệu ứng nhấn - lún xuống */
    .stButton button:active {
        transform: translateY(4px) !important;
        box-shadow: 0 2px 0 #000000 !important;
        transition: all 0.02s linear !important;
    }
    
    /* Focus - xóa viền */
    .stButton button:focus, 
    .stButton button:focus-visible {
        outline: none !important;
        box-shadow: 0 6px 0 #000000 !important;
        border: 2px solid #000000 !important;
    }
    
    .prediction-box {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 1rem;
        margin-top: 1.5rem;
        border-top: none;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .confidence-text {
        text-align: center;
        font-size: 0.75rem;
        font-weight: 500;
        color: #5f6368;
        margin-top: 0.5rem;
    }
    
    .prob-text {
        text-align: center;
        font-size: 0.7rem;
        font-weight: 400;
        color: #000000;
        margin-top: 1rem;
        line-height: 1.6;
        letter-spacing: 0.5px;
    }
    
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #e0e0e0;
    }
    
    /* Style cho canvas */
    .canvas-wrapper {
        display: flex;
        justify-content: center;
    }
    
    /* Style cho warning */
    .stAlert {
        font-family: 'Source Code Pro', monospace !important;
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">EMOJI CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">by MLP model v1.0 - 2026</div>', unsafe_allow_html=True)

# ------------------- TRAIN MODEL -------------------
@st.cache_resource
def load_and_train_model():
    if not os.path.exists("emoji-dataset"):
        with st.spinner("loading dataset..."):
            os.system('git clone https://github.com/thuanluong31241027504-sketch/emoji-dataset.git')
    
    data, labels = [], []
    emoji_path = 'emoji-dataset/my_emoji_dataset'
    
    for emoji_name in os.listdir(emoji_path):
        emoji_folder = os.path.join(emoji_path, emoji_name)
        if os.path.isdir(emoji_folder):
            for img_file in os.listdir(emoji_folder):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(emoji_folder, img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        data.append(cv2.resize(img, (28, 28)))
                        labels.append(emoji_name)
    
    X = np.array(data, dtype=np.uint8).astype('float32') / 255.0
    unique_labels = np.unique(labels)
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    y = to_categorical(np.array([label_to_id[l] for l in labels]))
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
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

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
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
    
    # Nút CONFIRM - hình vuông, block 3D
    if st.button("CONFIRM", key="confirm_btn"):
        if canvas_result.image_data is not None:
            if np.sum(canvas_result.image_data[:, :, 3]) > 100:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
                processed = preprocess_image(img)
                predictions = model.predict(processed, verbose=0)[0]
                st.session_state.last_probs = predictions
                predicted_idx = np.argmax(predictions)
                st.session_state.prediction = class_names[predicted_idx]
                st.session_state.confidence = predictions[predicted_idx]
            else:
                st.warning("draw something first")
        else:
            st.warning("draw something first")

if st.session_state.prediction:
    display_name = class_display.get(st.session_state.prediction, st.session_state.prediction.upper())
    st.markdown(f"""
    <div class="prediction-box">
        {display_name}
    </div>
    <div class="confidence-text">
        confidence: {st.session_state.confidence:.2%}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.last_probs is not None:
        prob_lines = []
        for i, name in enumerate(class_names):
            prob_lines.append(f"{name}: {st.session_state.last_probs[i]:.2%}")
        st.markdown(f'<div class="prob-text">{"  |  ".join(prob_lines)}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align: center; font-size: 0.7rem; color: #9aa0a6;">Source Code Pro · TensorFlow · MLP</div>', unsafe_allow_html=True)
