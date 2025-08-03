import streamlit as st
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_auc_score

# Load models
@st.cache_resource
def load_spade_model():
    model = torch.load("models/spade_model.pt", map_location="cpu")
    model.eval()
    return model

@st.cache_resource
def load_clip_data():
    clip_scores = np.load("npy_scores/clip_fused_score.npy")
    clip_labels = np.load("npy_scores/clip_test_labels.npy")
    return clip_scores, clip_labels

# Normalize and preprocess image
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# SPADE Heatmap
def generate_spade_heatmap(model, image_tensor):
    image_tensor = image_tensor.to("cpu")
    _ = model(image_tensor)
    cam = model.generate_fused_cam(image_tensor)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=4)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

# Visual Display
def show_heatmap_overlay(image, cam, title=""):
    img = np.array(image.resize((224, 224))) / 255.0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.axis("off")
    st.pyplot(plt)

# Streamlit UI
st.set_page_config(page_title="Anomaly Detection Demo", layout="wide")
st.title("Anomaly Detection: SPADE vs CLIP")

uploaded_image = st.file_uploader("Upload an image from the test set", type=["jpg", "png", "jpeg"])
demo_images = ["demo_images/" + f for f in ["image1.jpg", "image2.jpg"]]

if uploaded_image or st.button("Use Demo Image"):
    if uploaded_image:
        img = Image.open(uploaded_image)
        image_tensor = preprocess_image(uploaded_image)
    else:
        img = Image.open(demo_images[0])
        image_tensor = preprocess_image(demo_images[0])

    st.image(img, caption="Uploaded Image", width=300)

    # SPADE
    st.subheader(" SPADE Prediction")
    spade_model = load_spade_model()
    cam = generate_spade_heatmap(spade_model, image_tensor)
    show_heatmap_overlay(img, cam, "SPADE Heatmap")

    # CLIP
    st.subheader("CLIP Ensemble Prediction")
    clip_scores, clip_labels = load_clip_data()
    st.write(f"CLIP Score (simulated): `{clip_scores[0]:.4f}`")
    st.write(f"Prediction: {'Anomalous' if clip_scores[0] > 0.5 else 'Normal'}")

    # Combined Interpretation
    st.success("Prediction complete. Use both visual (SPADE) and score-based (CLIP) explanations to make your case.")
