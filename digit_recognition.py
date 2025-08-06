import streamlit as st
import os
import zipfile
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import tempfile

# 🌄 Add background image and custom styles
def add_bg_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}

        .custom-box {{
            background-color: rgba(255, 255, 255, 0.6);
            padding: 2rem;
            border-radius: 12px;
            color: #000;
        }}

        .predict-box {{
            background-color: rgba(255, 255, 255, 0.7);
            color: #111;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            width: fit-content;
            box-shadow: 0 0 8px rgba(0,0,0,0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Set background
st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")
add_bg_from_url("https://miro.medium.com/v2/resize:fit:1358/0*zP5ZflqXdht5v9lc.jpg")

# ✅ Title and subtitle
st.markdown("""
    <h1 style='text-align: center; color: #ffffff; text-shadow: 2px 2px 4px #000000;'>
        🧠 Handwritten Digit Recognizer
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; font-size: 18px; color: #ffffff; text-shadow: 1px 1px 2px #000000;'>
        Train a model with ZIP file (0–9 folders) and predict digits from images
    </p>
""", unsafe_allow_html=True)

# 🧰 Helper functions
def extract_zip(zip_file, extract_path):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def train_model(folder_path):
    X, y = [], []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (8, 8))
            X.append(img_resized.flatten())
            y.append(int(label))

    if not X:
        return None

    model = RandomForestClassifier()
    model.fit(np.array(X), np.array(y))
    return model

def predict_digit(model, image):
    img = image.convert("L")
    img = img.resize((8, 8))
    img_array = np.array(img)
    img_flat = img_array.flatten().reshape(1, -1)
    prediction = model.predict(img_flat)[0]
    return prediction

# 📄 Upload inputs
st.sidebar.header("📦 Upload Training ZIP")
zip_file = st.sidebar.file_uploader("ZIP file (with folders named 0–9)", type="zip")

st.sidebar.header("🖼 Upload Image for Prediction")
test_img = st.sidebar.file_uploader("Digit image to predict", type=["png", "jpg", "jpeg"])
# 🔄 Train if ZIP uploaded or Load pretrained model

model = None
model_path = "model/digit_model.pkl"

if zip_file:
    with st.spinner("Extracting ZIP and training model..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "data.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())
            extract_zip(zip_path, tmpdir)
            model = train_model(tmpdir)

            if model:
                os.makedirs("model", exist_ok=True)
                joblib.dump(model, model_path)

    if model:
        st.markdown("""
        <div style='
            background-color: rgba(220, 255, 220, 0.8);
            padding: 10px 20px;
            border-radius: 8px;
            color: green;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
        '>
        ✅ Model trained successfully!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Failed to train model. Check ZIP content (folders 0–9 with images).")

else:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("📦 Loaded pre-trained model from saved file. Now Upload image for prediction")
    else:
        st.warning("⚠️ No model found. Please upload training ZIP to train one.")


# ✅ Prediction
if test_img and model:
    st.markdown("""
        <h3 style='color: #ffffff; text-shadow: 1px 1px 2px #000000;'>
            🖼 Test Image
        </h3>
    """, unsafe_allow_html=True)

    image = Image.open(test_img)
    st.image(image, caption="", width=150)

    if st.button("🎯 Predict Digit"):
        with st.spinner("Predicting..."):
            prediction = predict_digit(model, image)
            st.markdown(f"<div class='predict-box'>🧠 Predicted Digit: {prediction}</div>", unsafe_allow_html=True)

elif test_img and not model:
    st.warning("⚠️ Upload and train the model first before predicting.")