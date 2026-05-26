import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas  # DÙNG COMPONENT CHUYÊN DỤNG
import os
import base64
from io import BytesIO

# ... (Phần config trang và hàm train_model giữ nguyên như bạn đã có) ...
st.set_page_config(page_title="Emoji Classifier", page_icon="🎨")
st.title("🎨 Nhận diện Emoji với AI")
st.markdown("*Vẽ trực tiếp lên khung bên dưới - AI sẽ nhận diện ngay lập tức!*")

@st.cache_resource
def load_and_train_model():
    # ... (code train model y hệt như bạn đã có, mình giữ nguyên) ...
    # Clone dataset...
    if not os.path.exists("emoji-dataset"):
        with st.spinner("Đang tải dataset..."):
            os.system('git clone https://github.com/thuanluong31241027504-sketch/emoji-dataset.git')
    # Đọc dữ liệu...
    data = []
    labels = []
    emoji_path = 'emoji-dataset/my_emoji_dataset'
    # ... (phần đọc ảnh) ...
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
    # ... (phần xử lý dữ liệu) ...
    X = np.array(data, dtype=np.uint8)
    y = np.array(labels)
    X = X.astype('float32') / 255.0
    unique_labels = np.unique(y)
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_to_id[label] for label in labels])
    y_categorical = to_categorical(y_numeric)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    # ... (phần xây dựng model) ...
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(unique_labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # ... (phần train) ...
    with st.spinner("Đang train model (lần đầu sẽ mất 2-3 phút)..."):
        model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, verbose=0)
    return model, unique_labels

def preprocess_image(image):
    # Resize và chuẩn hóa ảnh từ canvas
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

# === PHẦN QUAN TRỌNG: XỬ LÝ ĐẶC BIỆT ĐỂ HIỂN THỊ NỀN TRÊN CLOUD ===
icons = {'cloud': '☁️', 'grinning_face': '😀', 'heart': '❤️', 'smiling_horns': '👿', 'thumb': '👍'}
names = {'cloud': 'Cloud', 'grinning_face': 'Smiley Face', 'heart': 'Heart', 'smiling_horns': 'Horned Smiley', 'thumb': 'Thumbs Up'}

# Tạo một ảnh nền trắng, chuyển sang Base64 để làm background cho canvas
# Cách này giúp tránh lỗi mất ảnh nền khi deploy lên Streamlit Cloud
bg_image = Image.new("RGB", (400, 400), "white")
buffered = BytesIO()
bg_image.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()
bg_image_base64 = f"data:image/png;base64,{img_base64}"

# Tùy chọn vẽ
stroke_width = st.sidebar.slider("Độ dày nét vẽ: ", 5, 30, 15)
stroke_color = st.sidebar.color_picker("Màu vẽ: ", "#000000")
drawing_mode = st.sidebar.selectbox("Công cụ vẽ: ", ("freedraw", "line", "rect", "circle"))
st.sidebar.markdown("---")
st.sidebar.write("🎨 **Hướng dẫn:** Vẽ emoji lên khung trắng bên cạnh!")

# TẠO CANVAS VẼ CHUYÊN NGHIỆP
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # Màu fill trong suốt
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="White",  # Màu nền
    background_image=bg_image_base64,  # Truyền ảnh nền dạng Base64 để chắc chắn hoạt động
    update_streamlit=True,
    height=350,
    width=350,
    drawing_mode=drawing_mode,
    key="canvas",
)

# XỬ LÝ KHI NGƯỜI DÙNG VẼ XONG
if canvas_result.image_data is not None:
    # Lấy dữ liệu ảnh từ canvas
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
    # Tiền xử lý ảnh
    processed_img = preprocess_image(img)
    # Dự đoán
    predictions = model.predict(processed_img, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_label = class_names[predicted_idx]
    confidence = predictions[predicted_idx]
    
    # Hiển thị kết quả
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if confidence > 0.7:
            st.success(f"### {icons.get(predicted_label, '🎨')} AI Đoán là: **{names.get(predicted_label, predicted_label)}**")
            st.metric("Độ tin cậy", f"{confidence:.2%}")
        else:
            st.warning(f"### 🤔 AI chưa chắc lắm... Đoán là: **{names.get(predicted_label, predicted_label)}**")
            st.metric("Độ tin cậy", f"{confidence:.2%}")

st.markdown("---")
st.caption("💡 Mẹo: Vẽ đơn giản, rõ nét và **KHÔNG** vẽ ngược (nét đen trên nền trắng) để AI nhận diện tốt nhất!")
