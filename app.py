import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Emoji Classifier", page_icon="🎨")
st.title("🎨 Nhận diện Emoji với AI")
st.markdown("*Vẽ hoặc upload ảnh emoji - Hệ thống sẽ nhận diện*")

@st.cache_resource
def load_and_train_model():
    # Clone dataset
    if not os.path.exists("emoji-dataset"):
        with st.spinner("Đang tải dataset..."):
            os.system('git clone https://github.com/thuanluong31241027504-sketch/emoji-dataset.git')
    
    # Đọc dữ liệu
    data = []
    labels = []
    emoji_path = 'emoji-dataset/my_emoji_dataset'
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    emoji_folders = [f for f in os.listdir(emoji_path) if os.path.isdir(os.path.join(emoji_path, f))]
    
    for idx, emoji_name in enumerate(emoji_folders):
        emoji_folder = os.path.join(emoji_path, emoji_name)
        for img_file in os.listdir(emoji_folder):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(emoji_folder, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (28, 28))
                    data.append(img)
                    labels.append(emoji_name)
        progress_bar.progress((idx + 1) / len(emoji_folders))
        status_text.text(f"Đang đọc ảnh: {emoji_name}")
    
    status_text.text("Đang xử lý dữ liệu...")
    
    X = np.array(data, dtype=np.uint8)
    y = np.array(labels)
    
    # Tiền xử lý
    X = X.astype('float32') / 255.0
    unique_labels = np.unique(y)
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_to_id[label] for label in labels])
    y_categorical = to_categorical(y_numeric)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    # Xây dựng model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(unique_labels), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    status_text.text("Đang train model...")
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=0)
    
    # Lấy accuracy cuối cùng
    final_acc = history.history['accuracy'][-1]
    status_text.text(f"✅ Train xong! Accuracy: {final_acc:.2%}")
    
    progress_bar.empty()
    
    return model, unique_labels

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

# Load hoặc train model
model, class_names = load_and_train_model()

st.success(f"✅ Model đã sẵn sàng! Nhận diện {len(class_names)} loại emoji")

icons = {'cloud': '☁️', 'grinning_face': '😀', 'heart': '❤️', 'smiling_horns': '👿', 'thumb': '👍'}

tab1, tab2 = st.tabs(["✏️ Vẽ Emoji", "📤 Upload ảnh"])

with tab1:
    drawing = st.canvas("", width=280, height=280, background_color="#FFFFFF", 
                        stroke_color="#000000", stroke_width=15, key="canvas")
    if drawing.image_data is not None and st.button("🔍 Nhận diện", key="draw"):
        with st.spinner("Đang xử lý..."):
            img = Image.fromarray(drawing.image_data.astype('uint8'), mode='RGBA').convert('L')
            processed = preprocess_image(img)
            pred = model.predict(processed, verbose=0)[0]
            label = class_names[np.argmax(pred)]
            conf = max(pred)
            st.success(f"### {icons.get(label, '🎨')} Kết quả: {label}")
            st.metric("Độ tin cậy", f"{conf:.2%}")

with tab2:
    uploaded = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'])
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Ảnh của bạn", width=200)
        if st.button("🔍 Nhận diện", key="upload"):
            with st.spinner("Đang xử lý..."):
                processed = preprocess_image(img)
                pred = model.predict(processed, verbose=0)[0]
                label = class_names[np.argmax(pred)]
                conf = max(pred)
                st.success(f"### {icons.get(label, '🎨')} Kết quả: {label}")
                st.metric("Độ tin cậy", f"{conf:.2%}")

st.markdown("---")
st.caption(f"Model MLP - Train trên {len(data)} ảnh emoji")
