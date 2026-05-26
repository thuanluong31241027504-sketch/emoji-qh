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

# ==================== MÀN HÌNH GIỚI THIỆU ====================
if 'show_app' not in st.session_state:
    st.session_state.show_app = False

if not st.session_state.show_app:
    st.set_page_config(page_title="Emoji Classifier", layout="centered")
    
    # Đường dẫn ảnh
    decor_url = "https://raw.githubusercontent.com/thuanluong31241027504-sketch/emoji-qh/main/images/qq32.png"
    
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: #FFFFFF !important;
        }}
        
        .splash-wrapper {{
            max-width: 420px;
            margin: 80px auto 0 auto;
        }}
        
        .splash-box {{
            background-color: #000000;
            padding: 2rem;
            text-align: center;
            border-radius: 0px;
        }}
        
        .splash-text {{
            font-family: 'Source Code Pro', monospace;
            color: #FFFFFF;
            text-align: left;
            line-height: 1.8;
        }}
        
        .splash-desc {{
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
        }}
        
        .splash-rule {{
            font-size: 0.75rem;
            color: #AAAAAA;
            margin-top: 1rem;
        }}
        
        .splash-rule p {{
            margin: 0.3rem 0;
        }}
        
        .splash-wrapper button[kind="secondary"] {{
            background: #FFFFFF !important;
            color: #000000 !important;
            border: 2px solid #000000 !important;
            border-radius: 0px !important;
            padding: 10px 28px !important;
            font-family: 'Source Code Pro', monospace !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            cursor: pointer !important;
            box-shadow: 0 6px 0 #000000 !important;
            transition: all 0.05s linear !important;
            width: auto !important;
            min-width: 120px !important;
            margin-top: 1.5rem !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        
        .splash-wrapper button[kind="secondary"]:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 0 #000000 !important;
            background: #FAFAFA !important;
        }}
        
        .splash-wrapper button[kind="secondary"]:active {{
            transform: translateY(3px) !important;
            box-shadow: 0 3px 0 #000000 !important;
        }}
        
        .decor-top {{
            text-align: center;
            margin-bottom: 1rem;
        }}
        
        .decor-top img {{
            width: 48px;
            opacity: 0.6;
        }}
        
        #MainMenu, footer, header {{
            display: none !important;
        }}
    </style>
    
    <div class="splash-wrapper">
        <div class="decor-top">
            <img src="{decor_url}" alt="decor">
        </div>
        <div class="splash-box">
            <div class="splash-text">
                <div class="splash-desc">
                    Ứng dụng dự đoán emoji từ nét vẽ chuột của bạn một cách nhanh chóng và chính xác, được xây dựng trên kiến trúc MLP — phiên bản v1.0-2026.
                </div>
                <div class="splash-rule">
                    <p>> vẽ bất cứ thứ gì trong khung trắng</p>
                    <p>> nhấn CONFIRM</p>
                    <p>> nhận kết quả dự đoán từ model</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("START!"):
            st.session_state.show_app = True
            st.rerun()
    
    st.stop()

# ==================== APP CHÍNH ====================
st.set_page_config(page_title="Emoji Classifier", layout="centered")

# Đường dẫn ảnh hoa và mây
hoa1_url = "https://raw.githubusercontent.com/thuanluong31241027504-sketch/emoji-qh/main/images/hoa1.png"
hoa2_url = "https://raw.githubusercontent.com/thuanluong31241027504-sketch/emoji-qh/main/images/hoa2.png"
hoa3_url = "https://raw.githubusercontent.com/thuanluong31241027504-sketch/emoji-qh/main/images/hoa3.png"
may1_url = "https://raw.githubusercontent.com/thuanluong31241027504-sketch/emoji-qh/main/images/may1.png"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600;700&display=swap');
    
    * {{
        font-family: 'Source Code Pro', 'Courier New', monospace !important;
    }}
    
    .title {{
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }}
    
    .sub {{
        font-size: 0.8rem;
        text-align: center;
        color: #5f6368;
        margin-bottom: 2rem;
    }}
    
    /* Ảnh mây phía trên - góc trái */
    .cloud-decor {{
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 100;
        opacity: 0.5;
    }}
    
    .cloud-decor img {{
        width: 60px;
    }}
    
    /* Hàng hoa phía dưới */
    .flower-row {{
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin-top: 2rem;
        padding: 1rem 0;
        flex-wrap: wrap;
        gap: 10px;
    }}
    
    .flower-row img {{
        width: 35px;
        opacity: 0.5;
        transition: opacity 0.2s;
    }}
    
    .flower-row img:hover {{
        opacity: 0.8;
    }}
    
    #MainMenu, footer, header {{
        display: none !important;
    }}
    
    .stCanvas {{
        display: flex !important;
        justify-content: center !important;
    }}
    
    .stButton {{
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        height: 100% !important;
        margin-top: 70px !important;
    }}
    
    .stButton button {{
        background: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 40px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        font-family: 'Source Code Pro', monospace !important;
        cursor: pointer !important;
        box-shadow: 0 6px 0 #000000 !important;
        transition: none !important;
        white-space: nowrap !important;
    }}
    
    .stButton button:hover {{
        background: #FFFFFF !important;
        transform: none !important;
        box-shadow: 0 6px 0 #000000 !important;
    }}
    
    .stButton button:active {{
        transform: translateY(3px) !important;
        box-shadow: 0 3px 0 #000000 !important;
    }}
    
    .prediction {{
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 1rem;
        margin-top: 1.5rem;
        border-bottom: 2px solid #000;
    }}
    
    .confidence {{
        text-align: center;
        font-size: 0.75rem;
        color: #5f6368;
        margin-top: 0.5rem;
    }}
    
    .probs {{
        text-align: center;
        font-size: 0.7rem;
        margin-top: 1rem;
        line-height: 1.6;
    }}
</style>

<!-- Ảnh mây góc trái phía trên -->
<div class="cloud-decor">
    <img src="{may1_url}" alt="cloud">
</div>

<div class="title">EMOJI CLASSIFIER</div>
<div class="sub">by MLP model v1.0 - 2026</div>
""", unsafe_allow_html=True)

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

col_canvas, col_button = st.columns([2, 1])

with col_canvas:
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

with col_button:
    st.write("")
    st.write("")
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

if st.session_state.pred:
    st.markdown(f'<div class="prediction">{display.get(st.session_state.pred, st.session_state.pred.upper())}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">{st.session_state.conf:.2%}</div>', unsafe_allow_html=True)
    
    if st.session_state.probs is not None:
        text = "  |  ".join([f"{c}: {st.session_state.probs[i]:.2%}" for i, c in enumerate(classes)])
        st.markdown(f'<div class="probs">{text}</div>', unsafe_allow_html=True)

# ==================== HÀNG HOA PHÍA DƯỚI ====================
# Tạo chuỗi hoa xen kẽ: hoa1, hoa2, hoa3, hoa1, hoa2, hoa3...
flowers = [hoa1_url, hoa2_url, hoa3_url] * 8  # lặp 8 lần = 24 bông hoa

flower_html = '<div class="flower-row">'
for i, url in enumerate(flowers):
    flower_html += f'<img src="{url}" alt="flower">'
flower_html += '</div>'

st.markdown(flower_html, unsafe_allow_html=True)
