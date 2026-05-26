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
    
    #MainMenu, footer, header {
        display: none !important;
    }
    
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    /* === NÚT 3D BO GÓC MÀU XANH (cũ đã chạy) === */
    .pushable {
        background: #0d47a1;
        border-radius: 12px;
        border: none;
        padding: 0;
        cursor: pointer;
        outline-offset: 4px;
        margin-top: 20px;
    }
    
    .front {
        display: block;
        padding: 12px 32px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 700;
        background: #2196f3;
        color: white;
        transform: translateY(-6px);
        font-family: 'Source Code Pro', monospace !important;
    }
    
    .pushable:active .front {
        transform: translateY(-2px);
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
    
    .footer {
        text-align: center;
        font-size: 0.7rem;
        color: #9aa0a6;
        margin-top: 2rem;
    }
    
    /* Ẩn nút Streamlit mặc định */
    .stButton {
        display: none !important;
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
    st.session_state.clicked = False

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Canvas vẽ - khung nét đứt xám
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
    
    # Nút 3D bo góc màu xanh
    import streamlit.components.v1 as components
    
    components.html("""
    <div style="display: flex; justify-content: center;">
        <button class="pushable" id="confirmBtn">
            <span class="front">CONFIRM!</span>
        </button>
    </div>
    
    <script>
        document.getElementById('confirmBtn').addEventListener('click', () => {
            const event = new CustomEvent('streamlit:click', {
                detail: {button: 'confirm'}
            });
            window.parent.document.dispatchEvent(event);
        });
    </script>
    """, height=100)
    
    # Button ẩn để xử lý
    if st.button("", key="hidden_btn"):
        st.session_state.clicked = True
    
    if st.session_state.clicked:
        st.session_state.clicked = False
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
st.markdown('<div class="footer">Source Code Pro · TensorFlow · MLP</div>', unsafe_allow_html=True)
