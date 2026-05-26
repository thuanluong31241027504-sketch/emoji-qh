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
import matplotlib.pyplot as plt

# ============================================
# PHẦN 1: CẤU HÌNH TRANG (CSS)
# ============================================
st.set_page_config(page_title="Emoji Classifier", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Source Code Pro', 'Courier New', monospace !important; }
    
    .title { font-size: 2.5rem; font-weight: 700; text-align: center; margin-top: 1rem; }
    .sub { font-size: 0.8rem; text-align: center; color: #5f6368; margin-bottom: 2rem; }
    
    #MainMenu, footer, header { display: none !important; }
    
    .stCanvas { display: flex !important; justify-content: center !important; }
    
    .stButton {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        height: 100% !important;
        margin-top: 70px !important;
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
        white-space: nowrap !important;
    }
    
    .stButton button:hover {
        background: #FFFFFF !important;
        transform: none !important;
        box-shadow: 0 6px 0 #000000 !important;
    }
    
    .stButton button:active { transform: translateY(3px) !important; box-shadow: 0 3px 0 #000000 !important; }
    
    .prediction { text-align: center; font-size: 2rem; font-weight: 700; padding: 1rem; margin-top: 1.5rem; border-bottom: 2px solid #000; }
    .confidence { text-align: center; font-size: 0.75rem; color: #5f6368; margin-top: 0.5rem; }
    .probs { text-align: center; font-size: 0.7rem; margin-top: 1rem; line-height: 1.6; }
    
    .metric-box {
        background: #f8f9fa;
        border-left: 3px solid #000;
        padding: 0.8rem;
        margin: 1rem 0;
    }
    
    .metric-title { font-weight: 700; font-size: 0.8rem; }
    .metric-value { font-size: 1.2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ============================================
# PHẦN 2: HÀM TIỀN XỬ LÝ ẢNH
# ============================================
def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = 1.0 - img_array
    return img_array.reshape(1, 28, 28)

# ============================================
# PHẦN 3: MODEL - TRAIN VÀ ĐÁNH GIÁ
# ============================================
@st.cache_resource
def load_and_train_model():
    """Train model và trả về model, classes, cùng các chỉ số đánh giá"""
    
    with st.spinner("Đang tải dữ liệu..."):
        if not os.path.exists("emoji-dataset"):
            os.system('git clone https://github.com/thuanluong31241027504-sketch/emoji-dataset.git')
    
    # Đọc dữ liệu
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
    
    # Chia tập train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Xây dựng model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(unique), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train và lưu lịch sử
    with st.spinner("Đang huấn luyện model..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            verbose=0
        )
    
    # Đánh giá
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Dự đoán mẫu để tính F1-score
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(y_true_classes, y_pred_classes, target_names=unique, output_dict=True)
    
    metrics = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'history': history,
        'unique': unique,
        'report': report,
        'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes),
        'X_test': X_test,
        'y_test': y_test
    }
    
    return model, unique, metrics

# ============================================
# PHẦN 4: MÀN HÌNH GIỚI THIỆU (SPLASH)
# ============================================
if 'show_app' not in st.session_state:
    st.session_state.show_app = False

if not st.session_state.show_app:
    
    st.markdown("""
    <div style="max-width: 420px; margin: 100px auto 0 auto;">
        <div style="background-color: #000000; padding: 2rem;">
            <div style="font-family: 'Source Code Pro', monospace; color: #FFFFFF; text-align: left; line-height: 1.8;">
                <div style="font-size: 0.85rem; margin-bottom: 1.5rem;">
                    Ứng dụng dự đoán emoji từ nét vẽ chuột của bạn một cách nhanh chóng và chính xác, được xây dựng trên kiến trúc MLP — phiên bản v1.0-2026.
                </div>
                <div style="font-size: 0.75rem; color: #AAAAAA;">
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

# ============================================
# PHẦN 5: APP CHÍNH
# ============================================
st.markdown('<div class="title">EMOJI CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">by MLP model v1.0 - 2026</div>', unsafe_allow_html=True)

# Load model và metrics
model, classes, metrics = load_and_train_model()

# ============================================
# PHẦN 6: HIỂN THỊ THÔNG TIN MODEL
# ============================================
with st.expander("📊 Thông tin huấn luyện model", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">TRAIN ACCURACY</div>
            <div class="metric-value">{metrics['train_acc']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">VALIDATION ACCURACY</div>
            <div class="metric-value">{metrics['val_acc']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">TEST ACCURACY</div>
            <div class="metric-value">{metrics['test_acc']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Biểu đồ Loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(metrics['history'].history['loss'], label='Train Loss', color='black', linewidth=1.5)
    ax[0].plot(metrics['history'].history['val_loss'], label='Validation Loss', color='gray', linewidth=1.5)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Loss')
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(metrics['history'].history['accuracy'], label='Train Acc', color='black', linewidth=1.5)
    ax[1].plot(metrics['history'].history['val_accuracy'], label='Validation Acc', color='gray', linewidth=1.5)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].set_title('Accuracy')
    ax[1].grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # F1-score theo từng class
    st.markdown("**F1-score theo từng class:**")
    f1_data = []
    for class_name in classes:
        f1 = metrics['report'][class_name]['f1-score']
        f1_data.append(f1)
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(classes, f1_data, color='black', alpha=0.7)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('F1-score')
    ax2.set_title('F1-score cho từng class')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, f1_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2%}', ha='center', fontsize=9)
    
    st.pyplot(fig2)

# ============================================
# PHẦN 7: GIAO DIỆN CHÍNH
# ============================================
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

# Kết quả dự đoán
if st.session_state.pred:
    display_names = {
        'cloud': 'CLOUD',
        'grinning_face': 'SMILEY',
        'heart': 'HEART',
        'smiling_horns': 'HORNED',
        'thumb': 'THUMB'
    }
    st.markdown(f'<div class="prediction">{display_names.get(st.session_state.pred, st.session_state.pred.upper())}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">{st.session_state.conf:.2%}</div>', unsafe_allow_html=True)
    
    if st.session_state.probs is not None:
        text = "  |  ".join([f"{c}: {st.session_state.probs[i]:.2%}" for i, c in enumerate(classes)])
        st.markdown(f'<div class="probs">{text}</div>', unsafe_allow_html=True)
