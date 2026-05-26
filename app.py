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

# ==================== THAM SỐ CĂN CHỈNH ====================
# CHỈ CẦN SỬA 2 SỐ NÀY
LEFT_COL = 10      # Cột trái (tăng lên -> canvas sang phải)
RIGHT_COL = 1     # Cột phải (tăng lên -> canvas sang trái)
BUTTON_SHIFT = 55 # Dịch nút sang phải

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@300;400;500;600;700&display=swap');
    
    html, body, .stApp, div, p, span, h1, h2, h3, h4, button, label {{
        font-family: 'Source Code Pro', 'Courier New', monospace !important;
        background-color: #FFFFFF;
        color: #000000;
    }}
    
    .big-title {{
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0rem;
    }}
    
    .sub-text {{
        font-size: 0.8rem;
        font-weight: 400;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    .stActionButton, .stActionButton button, [data-testid="baseActionButton"] {{
        display: none !important;
    }}
    
    .stCanvasToolbar {{
        display: none !important;
    }}
    
    /* CHỈNH NÚT */
    .stButton {{
        display: flex !important;
        justify-content: center !important;
        margin-top: 1.5rem !important;
        margin-left: {BUTTON_SHIFT}px !important;
    }}
    
    .stButton button {{
        background: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 40px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        font-family: 'Source Code Pro', monospace !important;
        width: auto !important;
        min-width: 180px !important;
        cursor: pointer !important;
        box-shadow: 0 8px 0 #000000, 0 4px 12px rgba(0,0,0,0.1) !important;
        transition: none !important;
        letter-spacing: 0.5px !important;
    }}
    
    .stButton button:hover {{
        background: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        transform: none !important;
        box-shadow: 0 8px 0 #000000, 0 4px 12px rgba(0,0,0,0.1) !important;
    }}
    
    .stButton button:active {{
        transform: translateY(4px) !important;
        box-shadow: 0 4px 0 #000000, 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.02s linear !important;
    }}
    
    .stButton button:focus, 
    .stButton button:focus-visible {{
        outline: none !important;
        box-shadow: 0 8px 0 #000000, 0 4px 12px rgba(0,0,0,0.1) !important;
    }}
    
    .prediction-box {{
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 1rem;
        margin-top: 1.5rem;
        border-top: none;
        border-bottom: 1px solid #e0e0e0;
    }}
    
    .confidence-text {{
        text-align: center;
        font-size: 0.8rem;
        color: #5f6368;
        margin-top: 0.5rem;
    }}
    
    .prob-text {{
        text-align: center;
        font-size: 0.7rem;
        font-family: 'Source Code Pro', monospace;
        color: #000000;
        margin-top: 1rem;
        line-height: 1.6;
    }}
    
    hr {{
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #e0e0e0;
    }}
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
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = 1.0 - img_array
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
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'last_probs' not in st.session_state:
    st.session_state.last_probs = None

# DÙNG COLUMNS ĐỂ CĂN CANVAS - THAY ĐỔI 2 SỐ NÀY ĐỂ DỊCH
col_left, col_mid, col_right = st.columns([LEFT_COL, 2, RIGHT_COL])

with col_mid:
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
    
    if st.button("CONFIRM!", key="confirm_btn"):
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
        {st.session_state.confidence:.2%}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.last_probs is not None:
        prob_lines = []
        for i, name in enumerate(class_names):
            prob_lines.append(f"{name}: {st.session_state.last_probs[i]:.2%}")
        st.markdown(f'<div class="prob-text">{"  |  ".join(prob_lines)}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align: center; font-size: 0.65rem; color: #9aa0a6;">Source Code Pro · TensorFlow · MLP</div>', unsafe_allow_html=True)
