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

st.set_page_config(page_title="Emoji Classifier", page_icon="🎨")
st.title("🎨 Nhận diện Emoji với AI")
st.markdown("*Vẽ trực tiếp lên khung bên dưới - AI sẽ nhận diện ngay lập tức!*")

@st.cache_resource
def load_and_train_model():
    if not os.path.exists("emoji-dataset"):
        with st.spinner("Đang tải dataset..."):
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
    
    with st.spinner("Đang train model (lần đầu sẽ mất 2-3 phút)..."):
        model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=0)
    
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

model, class_names = load_and_train_model()
st.success(f"✅ Model đã sẵn sàng! Nhận diện {len(class_names)} loại emoji")

icons = {'cloud': '☁️', 'grinning_face': '😀', 'heart': '❤️', 'smiling_horns': '👿', 'thumb': '👍'}
names = {'cloud': 'Cloud', 'grinning_face': 'Smiley Face', 'heart': 'Heart', 'smiling_horns': 'Horned Smiley', 'thumb': 'Thumbs Up'}

# Sidebar cho tùy chọn vẽ
st.sidebar.markdown("### 🎨 Công cụ vẽ")
stroke_width = st.sidebar.slider("Độ dày nét vẽ", 5, 30, 15)
stroke_color = st.sidebar.color_picker("Màu vẽ", "#000000")
drawing_mode = st.sidebar.selectbox("Chế độ vẽ", ("freedraw", "line", "rect", "circle"))
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Mẹo:** Vẽ nét đen trên nền trắng, đơn giản và rõ ràng!")

# Tạo canvas - KHÔNG dùng background_image, chỉ dùng background_color
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",  # Chỉ dùng màu trắng đơn giản
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        key="canvas",
    )

# Xử lý kết quả vẽ
if canvas_result.image_data is not None:
    # Chuyển đổi ảnh từ canvas
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
    
    # Kiểm tra xem có vẽ gì không (không phải ảnh trắng hoàn toàn)
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    
    if np.mean(img_array) < 250:  # Có vẽ gì đó
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_label = class_names[predicted_idx]
        confidence = predictions[predicted_idx]
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            if confidence > 0.7:
                st.success(f"### {icons.get(predicted_label, '🎨')} **{names.get(predicted_label, predicted_label)}**")
                st.metric("Độ tin cậy", f"{confidence:.2%}")
            else:
                st.warning(f"### 🤔 **{names.get(predicted_label, predicted_label)}**")
                st.metric("Độ tin cậy", f"{confidence:.2%}")
    else:
        st.info("✏️ Hãy vẽ một emoji vào khung bên cạnh!")

# Hiển thị các mẫu
st.markdown("---")
st.markdown("### 📋 Các mẫu emoji có thể nhận diện:")

cols = st.columns(5)
sample_labels = ['grinning_face', 'heart', 'thumb', 'cloud', 'smiling_horns']
sample_icons = ['😀', '❤️', '👍', '☁️', '👿']
sample_names = ['Smiley Face', 'Heart', 'Thumbs Up', 'Cloud', 'Horned Smiley']

for i, (col, icon, name) in enumerate(zip(cols, sample_icons, sample_names)):
    with col:
        st.markdown(f"<div style='text-align: center; font-size: 40px;'>{icon}</div>", unsafe_allow_html=True)
        st.caption(name)

st.markdown("---")
st.caption("🎨 Vẽ trực tiếp lên canvas - AI nhận diện theo thời gian thực!")
