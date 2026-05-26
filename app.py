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

# ------------------- CẤU HÌNH TRANG (Phong cách Minimal) -------------------
st.set_page_config(page_title="Let's Draw! - Emoji Classifier", page_icon="✏️", layout="wide")

# CSS tùy chỉnh để giống phông chữ Google Sans và bố cục Minimal
st.markdown("""
<style>
    /* Import Google Fonts - Giống phong chữ Quick Draw */
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
    
    html, body, .stApp {
        font-family: 'Google Sans', 'Segoe UI', Roboto, sans-serif;
        background-color: #FFFFFF;
    }
    
    /* Header lớn, đậm, căn giữa */
    .big-title {
        font-family: 'Google Sans', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0rem;
        padding-top: 1rem;
        color: #202124;
        letter-spacing: -0.5px;
    }
    .sub-text {
        font-family: 'Google Sans', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    .description-text {
        font-family: 'Google Sans', sans-serif;
        font-size: 1rem;
        text-align: center;
        color: #3c4043;
        max-width: 600px;
        margin: 0 auto 1rem auto;
        line-height: 1.5;
    }
    /* Ẩn các thành phần thừa của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton button {
        font-family: 'Google Sans', sans-serif;
        font-weight: 500;
        background-color: #000000;
        color: white;
        border-radius: 100px;
        padding: 0.5rem 2rem;
        border: none;
        transition: 0.2s;
    }
    .stButton button:hover {
        background-color: #3c4043;
        color: white;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #e0e0e0;
    }
    /* Style cho Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #f8f9fa;
    }
    .sidebar-text {
        font-family: 'Google Sans', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown('<div class="big-title">✏️ Let’s Draw!</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Can a neural network learn to recognize doodling?</div>', unsafe_allow_html=True)
st.markdown('<div class="description-text">Help teach it by adding your drawings. <br> AI will try to guess what you\'re drawing in real-time.</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------- HÀM TRAIN MODEL (GIỮ NGUYÊN) -------------------
@st.cache_resource
def load_and_train_model():
    with st.spinner("🎨 Training AI on emoji dataset... (Lần đầu sẽ mất 2-3 phút)"):
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

# ------------------- TIỀN XỬ LÝ ẢNH -------------------
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

# Load Model
model, class_names = load_and_train_model()

# ------------------- GIAO DIỆN VẼ CHÍNH (TRỌNG TÂM) -------------------
icons = {'cloud': '☁️', 'grinning_face': '😀', 'heart': '❤️', 'smiling_horns': '👿', 'thumb': '👍'}
names = {'cloud': 'Cloud', 'grinning_face': 'Smiley Face', 'heart': 'Heart', 'smiling_horns': 'Horned Smiley', 'thumb': 'Thumbs Up'}

# Sidebar nhỏ gọn
with st.sidebar:
    st.markdown("### 🎨 Settings")
    stroke_width = st.slider("Pen size", 5, 30, 15)
    stroke_color = st.color_picker("Pen color", "#000000")
    drawing_mode = st.selectbox("Tool", ("freedraw", "line", "rect", "circle"))
    st.markdown("---")
    st.markdown("#### 🧠 AI can recognize:")
    cols = st.columns(2)
    for i, name in enumerate(names.keys()):
        if i%2==0:
            cols[0].markdown(f"{icons.get(name, '')} {names.get(name)}")
        else:
            cols[1].markdown(f"{icons.get(name, '')} {names.get(name)}")

# Layout chính: Khung vẽ to ở giữa
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div style="text-align: center; font-weight: 500; margin-bottom: 10px;">✨ Draw something here ✨</div>', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        update_streamlit=True,
        height=350,
        width=350,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    
    # Nút Clear (giữ đúng phong cách Minimal)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        clear = st.button("🗑️ Clear Canvas", use_container_width=True)

# Xử lý clear canvas (dùng session state)
if 'clear' not in st.session_state:
    st.session_state.clear = False
if clear:
    st.session_state.clear = True
    st.rerun()
if st.session_state.clear:
    canvas_result = None
    st.session_state.clear = False

# ------------------- KẾT QUẢ DỰ ĐOÁN -------------------
if canvas_result is not None and canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
    img_gray = img.convert('L')
    
    if np.mean(np.array(img_gray)) < 250:
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_label = class_names[predicted_idx]
        confidence = predictions[predicted_idx]
        
        # Hiển thị kết quả to, rõ ràng, phong cách "Quick Draw"
        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns([1, 1.5, 1])
        with col_r2:
            if confidence > 0.6:
                st.markdown(f'<div style="text-align: center; font-size: 2.5rem; font-weight: 700;">🎨 I guess: <span style="background-color: #f1f3f4; padding: 0.2rem 1rem; border-radius: 50px;">{icons.get(predicted_label, "")} {names.get(predicted_label, predicted_label)}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align: center; color: #34a853; font-weight: 500;">Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: center; font-size: 2rem; font-weight: 700;">🤔 Hmm... maybe: <span style="background-color: #f1f3f4; padding: 0.2rem 1rem; border-radius: 50px;">{icons.get(predicted_label, "")} {names.get(predicted_label, predicted_label)}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align: center; color: #ea4335;">I\'m not very sure ({confidence:.2%})... Try drawing clearer!</div>', unsafe_allow_html=True)
    else:
        st.info("✏️ Start drawing in the white box above!")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #5f6368; font-size: 0.8rem;">Shared publicly to help with machine learning research. | Powered by TensorFlow & Streamlit</div>', unsafe_allow_html=True)
