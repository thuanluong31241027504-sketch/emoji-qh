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
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600;700;800&display=swap');
    
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
    
    #MainMenu, footer, header, .stActionButton, .stCanvasToolbar {
        display: none !important;
    }
    
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* === NÚT 3D BLOCK ĐÚNG CHUẨN === */
    .block-btn {
        position: relative;
        display: inline-block;
        background: none;
        border: none;
        cursor: pointer;
        margin-top: 20px;
    }
    
    /* Mặt chính (front face) */
    .block-btn .front {
        display: block;
        padding: 12px 28px;
        background-color: #FFFFFF;
        color: #000000;
        font-size: 16px;
        font-weight: 800;
        text-decoration: none;
        border: 2px solid #000000;
        transition: all 0.08s linear;
        position: relative;
        z-index: 2;
    }
    
    /* Mặt dưới (bottom face) - tạo chiều dày khối */
    .block-btn .bottom {
        position: absolute;
        bottom: -8px;
        left: 0px;
        width: 100%;
        height: 8px;
        background-color: #666666;
        border-left: 2px solid #000000;
        border-right: 2px solid #000000;
        border-bottom: 2px solid #000000;
        box-sizing: border-box;
        z-index: 1;
    }
    
    /* Mặt phải (right face) */
    .block-btn .right {
        position: absolute;
        top: 0px;
        right: -8px;
        width: 8px;
        height: 100%;
        background-color: #888888;
        border-top: 2px solid #000000;
        border-right: 2px solid #000000;
        border-bottom: 2px solid #000000;
        box-sizing: border-box;
        z-index: 1;
    }
    
    /* Hiệu ứng hover - nhấc lên */
    .block-btn:hover .front {
        transform: translate(-2px, -2px);
    }
    
    .block-btn:hover .bottom {
        bottom: -10px;
        height: 10px;
    }
    
    .block-btn:hover .right {
        right: -10px;
        width: 10px;
    }
    
    /* Hiệu ứng click - lún xuống */
    .block-btn:active .front {
        transform: translate(3px, 3px);
    }
    
    .block-btn:active .bottom {
        bottom: -3px;
        height: 3px;
    }
    
    .block-btn:active .right {
        right: -3px;
        width: 3px;
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
    
    # Nút 3D Block bằng HTML/CSS
    import streamlit.components.v1 as components
    
    components.html(f"""
    <div style="display: flex; justify-content: center; margin-top: 20px;">
        <button class="block-btn" id="btn_confirm">
            <span class="front">CONFIRM!</span>
            <span class="bottom"></span>
            <span class="right"></span>
        </button>
    </div>
    
    <script>
        document.getElementById('btn_confirm').addEventListener('click', () => {{
            const canvasData = window.parent.document.querySelector('.stCanvas').getAttribute('data');
            // Gửi sự kiện lên Streamlit
            window.parent.postMessage({{type: 'streamlit:setComponentValue', value: true}}, '*');
        }});
    </script>
    """, height=100)
    
    # Streamlit button để xử lý logic (ẩn)
    col_placeholder = st.empty()
    if col_placeholder.button("CONFIRM!", key="real_btn", use_container_width=False):
        if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:, :, 3]) > 100:
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
            pred = model.predict(preprocess_image(img), verbose=0)[0]
            st.session_state.last_probs = pred
            st.session_state.prediction = class_names[np.argmax(pred)]
            st.session_state.confidence = max(pred)
            st.rerun()
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
