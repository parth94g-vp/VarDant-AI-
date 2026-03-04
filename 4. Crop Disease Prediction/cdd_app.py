# # import streamlit as st
# # from ultralytics import YOLO
# # from PIL import Image
# # import numpy as np
# # import os

# # # ---------------- CONFIG ----------------
# # st.set_page_config(
# #     page_title="Crop Disease Detection 🌱",
# #     page_icon="🌿",
# #     layout="centered"
# # )

# # MODEL_PATH = r"D:\MIT\SEM 2\MP\Crop Detection Model\yolov8m_tuned\weights\best.pt"

# # # ---------------- LOAD MODEL ----------------
# # @st.cache_resource
# # def load_model():
# #     return YOLO(MODEL_PATH)

# # model = load_model()

# # # ---------------- UI HEADER ----------------
# # st.markdown(
# #     """
# #     <h1 style="text-align:center;">🌾 Crop Disease Detection System</h1>
# #     <p style="text-align:center; color:gray;">
# #         Upload a crop leaf image to detect diseases using YOLOv8
# #     </p>
# #     """,
# #     unsafe_allow_html=True
# # )

# # st.divider()

# # # ---------------- IMAGE UPLOAD ----------------
# # uploaded_file = st.file_uploader(
# #     "📤 Upload Crop Leaf Image",
# #     type=["jpg", "jpeg", "png"]
# # )

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file).convert("RGB")

# #     col1, col2 = st.columns([1, 1])

# #     with col1:
# #         st.image(image, caption="📸 Uploaded Image", use_container_width=True)

# #     with col2:
# #         st.markdown("### 🔍 Prediction Result")

# #         if st.button("🌿 Detect Disease"):
# #             with st.spinner("Analyzing leaf image..."):
# #                 img_array = np.array(image)
# #                 results = model(img_array)

# #                 if results[0].probs is not None:
# #                     cls_id = results[0].probs.top1
# #                     confidence = results[0].probs.top1conf.item()
# #                     class_name = model.names[cls_id]

# #                     st.success(f"**🌱 Disease Detected:** {class_name}")
# #                     st.info(f"**📊 Confidence:** {round(confidence * 100, 2)} %")

# #                     st.progress(min(confidence, 1.0))

# #                 else:
# #                     st.error("❌ No disease detected. Try another image.")

# # else:
# #     st.info("👆 Please upload a crop leaf image to begin.")

# # st.divider()

# # # ---------------- FOOTER ----------------
# # st.markdown(
# #     """
# #     <div style="text-align:center; color:gray; font-size:14px;">
# #         🚜 AI-powered Agriculture | YOLOv8 | Streamlit  
# #         <br>
# #         Developed for Mini Project
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )






# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import os

# # ===================== CONFIG =====================
# MODEL_PATH = r"D:\MIT\SEM 2\MP\Test_Project_1\best_model.pth"
# CONF_THRESHOLD = 0.60

# CLASS_NAMES = [
#     "Apple_Healthy", "Apple_Rust", "Apple_Scab",
#     "Blueberry_Healthy", "Cherry_Healthy", "Corn_Rust",
#     "Grape_Black_Rot", "Grape_Healthy", "Peach_Healthy",
#     "Pepper_Bacterial_Spot", "Pepper_Healthy",
#     "Potato_Early_Blight", "Potato_Healthy", "Potato_Late_Blight",
#     "Raspberry_Healthy", "Soybean_Healthy",
#     "Squash_Powdery_Mildew", "Strawberry_Healthy",
#     "Tomato_Bacterial_Spot", "Tomato_Early_Blight",
#     "Tomato_Late_Blight", "Tomato_Leaf_Mold",
#     "Tomato_Mosaic_Virus", "Tomato_Spider_Mites",
#     "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus"
# ]
# NUM_CLASSES = len(CLASS_NAMES)
# # ==================================================


# # ===================== MODEL =====================
# @st.cache_resource
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = models.efficientnet_v2_s(weights=None)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device)
#     model.eval()

#     return model, device


# def predict(image, model, device):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

#     img = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(img)
#         probs = torch.softmax(outputs, dim=1)

#     return probs.cpu().numpy()[0]


# # ===================== PAGE =====================
# st.set_page_config(
#     page_title="LeafCare – Crop Disease Detection",
#     page_icon="🌿",
#     layout="centered"
# )

# # ===================== SIDEBAR =====================
# st.sidebar.title("🌿 LeafCare")
# st.sidebar.write(
#     "Detect crop diseases from leaf images and get quick insights."
# )

# st.sidebar.markdown("### 📸 How to get best results")
# st.sidebar.markdown("""
# • Take a **clear photo** of one leaf  
# • Use **natural light**  
# • Avoid shadows & blur  
# • Keep the leaf **centered**
# """)

# st.sidebar.markdown("### ⚠️ Note")
# st.sidebar.write(
#     "This tool helps identify possible diseases. "
#     "For treatment decisions, consult an agriculture expert."
# )

# # ===================== MAIN =====================
# st.title("🌱 Crop Disease Detection")
# st.subheader("Upload a leaf image to check plant health")

# if not os.path.exists(MODEL_PATH):
#     st.error("❌ Model not found. Please contact support.")
#     st.stop()

# model, device = load_model()

# uploaded_file = st.file_uploader(
#     "📤 Upload leaf image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, use_container_width=True)

#     st.markdown("")

#     if st.button("🔍 Analyze Leaf", use_container_width=True):
#         with st.spinner("Analyzing leaf condition..."):
#             probs = predict(image, model, device)

#         top_idx = probs.argsort()[-3:][::-1]
#         top_label = CLASS_NAMES[top_idx[0]].replace("_", " ")
#         top_conf = probs[top_idx[0]]

#         st.markdown("---")

#         if top_conf < CONF_THRESHOLD:
#             st.warning(
#                 "⚠️ Image unclear. Please upload a clearer photo of the leaf."
#             )
#         else:
#             st.success("🌿 Analysis Complete")
#             st.markdown(f"### 🦠 Detected Condition: **{top_label}**")
#             st.progress(float(top_conf))
#             st.caption(f"Confidence level: {top_conf:.2f}")

#         st.markdown("### 📊 Other Possible Results")
#         for i in top_idx[1:]:
#             st.write(
#                 f"• {CLASS_NAMES[i].replace('_', ' ')} — {probs[i]:.2f}"
#             )

# # ===================== FOOTER =====================
# st.markdown("---")
# st.caption(
#     "LeafCare helps in early detection of crop diseases using image analysis."
# )











import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# ============================================================
# BASE PATH (IMPORTANT)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
CONF_THRESHOLD = 0.60

CLASS_NAMES = [
    "Apple_Healthy", "Apple_Rust", "Apple_Scab",
    "Blueberry_Healthy", "Cherry_Healthy", "Corn_Rust",
    "Grape_Black_Rot", "Grape_Healthy", "Peach_Healthy",
    "Pepper_Bacterial_Spot", "Pepper_Healthy",
    "Potato_Early_Blight", "Potato_Healthy", "Potato_Late_Blight",
    "Raspberry_Healthy", "Soybean_Healthy",
    "Squash_Powdery_Mildew", "Strawberry_Healthy",
    "Tomato_Bacterial_Spot", "Tomato_Early_Blight",
    "Tomato_Late_Blight", "Tomato_Leaf_Mold",
    "Tomato_Mosaic_Virus", "Tomato_Spider_Mites",
    "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus"
]

# ============================================================
# UI LABELS (MULTILINGUAL HEADINGS)
# ============================================================
UI_LABELS = {
    "English": {
        "info": "❓ Disease Information",
        "why": "🤔 Why it happens",
        "spread": "🌬️ How it spreads",
        "conditions": "🌦️ Favorable conditions",
        "symptoms": "🩺 Symptoms",
        "prevention": "🛡️ Prevention",
        "treatment": "💊 Treatment"
    },
    "Hindi": {
        "info": "❓ रोग की जानकारी",
        "why": "🤔 यह क्यों होता है",
        "spread": "🌬️ यह कैसे फैलता है",
        "conditions": "🌦️ अनुकूल परिस्थितियाँ",
        "symptoms": "🩺 लक्षण",
        "prevention": "🛡️ बचाव",
        "treatment": "💊 उपचार"
    },
    "Marathi": {
        "info": "❓ रोगाची माहिती",
        "why": "🤔 हा रोग का होतो",
        "spread": "🌬️ हा रोग कसा पसरतो",
        "conditions": "🌦️ अनुकूल परिस्थिती",
        "symptoms": "🩺 लक्षणे",
        "prevention": "🛡️ प्रतिबंध",
        "treatment": "💊 उपचार"
    }
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LeafCare – Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device

def predict(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

    return probs.cpu().numpy()[0]

# ============================================================
# LANGUAGE FILE HANDLING
# ============================================================
def validate_keys():
    try:
        en = json.load(open(os.path.join(BASE_DIR, "disease_info_eng.json"), encoding="utf-8"))
        hi = json.load(open(os.path.join(BASE_DIR, "disease_info_hin.json"), encoding="utf-8"))
        mr = json.load(open(os.path.join(BASE_DIR, "disease_info_mar.json"), encoding="utf-8"))

        if set(en.keys()) != set(hi.keys()) or set(en.keys()) != set(mr.keys()):
            st.warning("⚠️ Language files have mismatched disease entries.")
    except Exception:
        st.warning("⚠️ Unable to validate language files.")

@st.cache_data
def load_disease_data(language):
    file_map = {
        "English": "disease_info_eng.json",
        "Hindi": "disease_info_hin.json",
        "Marathi": "disease_info_mar.json"
    }

    try:
        with open(os.path.join(BASE_DIR, file_map[language]), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        with open(os.path.join(BASE_DIR, "disease_info_eng.json"), encoding="utf-8") as f:
            st.warning("⚠️ Selected language not available. Showing English.")
            return json.load(f)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🌿 LeafCare")
st.sidebar.write("AI-based crop disease detection system")

selected_language = st.sidebar.selectbox(
    "🌐 Select Language",
    ["English", "Hindi", "Marathi"]
)

labels = UI_LABELS[selected_language]

st.sidebar.markdown("### 📸 Image Tips")
st.sidebar.markdown("""
• Take a clear leaf photo  
• Use natural light  
• Avoid blur & shadows  
• Keep leaf centered
""")

# ============================================================
# LOAD DATA
# ============================================================
validate_keys()
DISEASE_INFO = load_disease_data(selected_language)

# ============================================================
# MAIN UI
# ============================================================
st.title("🌱 Crop Disease Detection")
st.subheader("Upload a leaf image to check plant health")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.stop()

model, device = load_model()

uploaded_file = st.file_uploader(
    "📤 Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("🔍 Analyze Leaf", use_container_width=True):
        with st.spinner("Analyzing leaf..."):
            probs = predict(image, model, device)

        idx = probs.argmax()
        label = CLASS_NAMES[idx]
        confidence = probs[idx]

        st.markdown("---")

        if confidence < CONF_THRESHOLD:
            st.warning("⚠️ Image unclear. Please upload a clearer leaf image.")
        else:
            info = DISEASE_INFO.get(label)

            if info:
                st.success("🌿 Analysis Complete")
                st.markdown(f"## 🦠 {info['disease_name']}")
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence:.2f}")

                with st.expander(labels["info"]):
                    st.write(info["what_is_it"])

                with st.expander(labels["why"]):
                    st.write(info["why_it_happens"])

                with st.expander(labels["spread"]):
                    st.write(info["how_it_spreads"])

                with st.expander(labels["conditions"]):
                    st.write(info["favorable_conditions"])

                with st.expander(labels["symptoms"]):
                    for s in info["symptoms"]:
                        st.write("•", s)

                with st.expander(labels["prevention"]):
                    for p in info["prevention"]:
                        st.write("•", p)

                with st.expander(labels["treatment"]):
                    for t in info["treatment"]:
                        st.write("•", t)

                st.success(f"👨‍🌾 {info['farmer_advice']}")
            else:
                st.warning("Disease information not available.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("LeafCare – Helping farmers detect crop diseases early using AI.")
