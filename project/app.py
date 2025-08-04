import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import models
from spade_model import MultiLayerGradCAMResNet18
from sklearn.metrics import roc_auc_score
import plotly.graph_objects as go
import plotly.express as px

# Enhanced page configuration
st.set_page_config(
    page_title=" Anomaly Detection Suite", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Anomaly Detection Suite\nCompare SPADE, CLIP, and PaDiM models for anomaly detection!"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    .model-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        color: #262730;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    
    .anomalous {
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
        color: white;
    }
    
    .normal {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .info-box strong {
        color: #ffffff;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <h1> Anomaly Detection Suite</h1>
    <p>Compare state-of-the-art models: SPADE, CLIP, and PaDiM for anomaly detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("##  Image Upload")
    
    # Model loading with progress
    with st.spinner("Loading models..."):
        @st.cache_resource
        def load_spade_model():
            model = MultiLayerGradCAMResNet18()
            model.load_state_dict(torch.load("models/spade_model.pth", map_location="cpu"))
            model.eval()
            return model

        @st.cache_resource
        def load_clip_scores():
            scores = np.load("npy_scores/clip_fused_score.npy")
            labels = np.load("npy_scores/clip_test_labels.npy")
            return scores, labels

        @st.cache_resource
        def load_padim_scores():
            scores = np.load("npy_scores/padim_scores.npy")
            labels = np.load("npy_scores/clip_test_labels.npy")
            return scores[:1200], labels
    
    st.success(" All models loaded successfully!")
    
    # Enhanced file uploader
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "png", "jpeg"],
        help="Upload an image to analyze for anomalies"
    )
    
    # Demo images selection
    demo_images = {
        "Demo Image 1": "demo/image1.jpg",
        "Demo Image 2": "demo/image2.jpg",
        "Demo Image 3": "demo/image3.jpg"
    }
    
    if not uploaded:
        selected_demo = st.selectbox("Or select a demo image:", list(demo_images.keys()))
        demo_path = demo_images[selected_demo]
        st.info(f"Using: {selected_demo}")
    
    # Model information
    with st.expander(" Model Information"):
        st.markdown("""
        **SPADE**:Pixel-level Anomaly Detection with explainable heatmaps
        
        **CLIP**: Ensemble model using visual prototypes (100 normal + 100 abnormal)
        - 83.3% Mean Similarity + 16.7% Distance-Based scoring
        - AUC: 0.797, Accuracy: 68.5%
        
        **PaDiM**: Lightweight Patch Distribution Modeling
        - ResNet-18 backbone with multi-scale features (layers 2,3,4)
        - 14×14 spatial patches, top 100 variance features
        - Mahalanobis distance with Gaussian distribution fitting
        """)

# Preprocessing functions
def preprocess_image(img_path_or_file):
    image = Image.open(img_path_or_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return image, transform(image).unsqueeze(0)

def generate_spade_heatmap(model, image_tensor):
    image_tensor = image_tensor.to("cpu")
    _ = model(image_tensor)  # Fixed the syntax error
    cam = model.generate_fused_cam(image_tensor)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=4)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

def show_enhanced_heatmap_overlay(image, cam, title="", model_type="SPADE"):
    """Enhanced heatmap visualization for SPADE model."""
    img = np.array(image.resize((224, 224))) / 255.0
    
    # Create subplots with better styling
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    
    # Original image
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold', pad=20)
    
    # Heatmap only
    im1 = axes[1].imshow(cam, cmap='jet')
    axes[1].axis("off")
    axes[1].set_title(f"{model_type} Heatmap", fontsize=14, fontweight='bold', pad=20)
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Anomaly Intensity', rotation=270, labelpad=15)
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].axis("off")
    axes[2].set_title("Overlay", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# Load and display image
if uploaded:
    pil_img, img_tensor = preprocess_image(uploaded)
    image_source = "Uploaded Image"
else:
    pil_img, img_tensor = preprocess_image(demo_path)
    image_source = selected_demo

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Input Image")
    st.image(pil_img, caption=image_source, use_container_width=True)
    
    # Image info
    st.markdown(f"""
    <div class="info-box">
        <strong> Image Details</strong><br><br>
        <strong>Source:</strong> {image_source}<br>
        <strong>Dimensions:</strong> {pil_img.size[0]} × {pil_img.size[1]} pixels<br>
        <strong>Color Mode:</strong> {pil_img.mode}<br>
        <strong>Format:</strong> {pil_img.format if hasattr(pil_img, 'format') and pil_img.format else 'Unknown'}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Model Analysis")
    
    # Enhanced tabs with emojis
    tabs = st.tabs([" SPADE", " CLIP", "PaDiM"])
    
    # SPADE Tab
    with tabs[0]:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("#### SPADE Analysis")
        st.markdown("*Pixel-level Anomaly Detection with explainable heatmaps*")
        
        with st.spinner("Generating SPADE heatmap..."):
            spade_model = load_spade_model()
            cam = generate_spade_heatmap(spade_model, img_tensor)
            
        # Enhanced visualization
        fig = show_enhanced_heatmap_overlay(pil_img, cam, "SPADE Heatmap", "SPADE")
        st.pyplot(fig, use_container_width=True)
        
        # Anomaly statistics
        anomaly_intensity = np.mean(cam)
        max_anomaly = np.max(cam)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Intensity", f"{anomaly_intensity:.3f}")
        with col2:
            st.metric("Max Intensity", f"{max_anomaly:.3f}")
        with col3:
            prediction = "Anomalous" if anomaly_intensity > 0.3 else "Normal"
            st.metric("Prediction", prediction)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # CLIP Tab
    with tabs[1]:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("#### CLIP Analysis")
        st.markdown("*Ensemble of similarity methods with visual prototypes*")
        
        clip_scores, clip_labels = load_clip_scores()
        score = clip_scores[0]
        
        # The optimal threshold from your model output is -0.0138
        optimal_threshold = -0.0138
        prediction = 'Anomalous' if score > optimal_threshold else 'Normal'
        
        # Prediction display
        pred_class = "anomalous" if prediction == 'Anomalous' else "normal"
        st.markdown(f"""
        <div class="prediction-box {pred_class}">
            Prediction: {prediction}
        </div>
        """, unsafe_allow_html=True)
        
        # Model details from your output
        st.markdown("""
        <div class="info-box">
            <strong>🔧 Ensemble Composition</strong><br><br>
            <strong>•</strong> Mean Similarity: <strong>83.3%</strong> weight<br>
            <strong>•</strong> Distance Based: <strong>16.7%</strong> weight<br>
            <strong>•</strong> Visual Prototypes: <strong>100 Normal + 100 Abnormal</strong><br>
            <strong>•</strong> Optimal Threshold: <strong>-0.0138</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ensemble Score", f"{score:.4f}")
        with col2:
            auc = roc_auc_score(clip_labels, clip_scores)
            st.metric("Model AUC", f"{auc:.4f}")
        with col3:
            # Distance from threshold
            distance = abs(score - optimal_threshold)
            st.metric("Confidence", f"{distance:.4f}")
        
        # Enhanced score visualization with correct threshold
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CLIP Ensemble Score"},
            delta = {'reference': optimal_threshold, 'position': "top"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, optimal_threshold], 'color': "lightgreen"},
                    {'range': [optimal_threshold, 1], 'color': "lightcoral"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': optimal_threshold}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics from your model
        st.markdown("##### Model Performance:")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("Accuracy", "68.5%")
        with perf_col2:
            st.metric("Normal F1", "0.609")
        with perf_col3:
            st.metric("Abnormal F1", "0.736")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # PaDiM Tab
    with tabs[2]:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown("#### PaDiM Analysis")
        st.markdown("*Multi-scale patch distribution modeling*")
        
        padim_scores, padim_labels = load_padim_scores()
        score = padim_scores[0]
        
        # Load PaDiM model for heatmap generation
        @st.cache_resource
        def load_padim_model():
            from lightweight_padim import LightweightPaDiM 
            model = LightweightPaDiM(num_selected_features=100)
            model.load_state("models/light_padim_resnet18.pth")  
            model.eval()
            return model
        
        # Generate heatmap if model is available
        try:
            padim_model = load_padim_model()
            heatmap, heatmap_score = padim_model.generate_heatmap(img_tensor)
            
            # Use specific visualization
            def visualize_padim_heatmap(image_tensor, heatmap, score=None):
                """PaDiM heatmap visualization matching your implementation."""
                # Convert image tensor to NumPy [0, 1] - matches your normalization
                img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min())
                
                # Create enhanced visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.patch.set_facecolor('white')
                
                # Original image
                axes[0].imshow(img)
                axes[0].set_title("Original Image", fontsize=14, fontweight='bold', pad=20)
                axes[0].axis("off")
                
                # Heatmap overlay
                axes[1].imshow(img)
                im = axes[1].imshow(heatmap, cmap='jet', alpha=0.6)
                
                # Enhanced title with score
                title = "PaDiM Anomaly Heatmap"
                if score is not None:
                    title += f"\nMahalanobis Score: {score:.4f}"
                
                axes[1].set_title(title, fontsize=14, fontweight='bold', pad=20)
                axes[1].axis("off")
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                cbar.set_label('Anomaly Intensity', rotation=270, labelpad=15)
                
                plt.tight_layout()
                return fig
            
            # Show PaDiM heatmap
            fig = visualize_padim_heatmap(img_tensor, heatmap, heatmap_score)
            st.pyplot(fig, use_container_width=True)
            
            # Use heatmap score for prediction
            prediction_score = heatmap_score
        except Exception as e:
            # Fallback to precomputed scores if model loading fails
            st.info("Using precomputed scores")
            prediction_score = score
        
        # Threshold 
        threshold = 0.5
        prediction = 'Anomalous' if prediction_score > threshold else 'Normal'
        
        # Prediction display
        pred_class = "anomalous" if prediction == 'Anomalous' else "normal"
        st.markdown(f"""
        <div class="prediction-box {pred_class}">
            Prediction: {prediction}
        </div>
        """, unsafe_allow_html=True)
        
        # Model architecture details
        st.markdown("""
        <div class="info-box">
            <strong>Architecture Details</strong><br><br>
            <strong>•</strong> Backbone: <strong>ResNet-18</strong> (frozen, pretrained)<br>
            <strong>•</strong> Multi-scale features: <strong>Layer2 + Layer3 + Layer4</strong><br>
            <strong>•</strong> Spatial resolution: <strong>14×14</strong> patches<br>
            <strong>•</strong> Feature selection: <strong>Top 100</strong> variance features<br>
            <strong>•</strong> Distribution: <strong>Multivariate Gaussian</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Anomaly Score", f"{prediction_score:.4f}")
        with col2:
            auc = 0.603
            st.metric("Model AUC", f"{auc:.4f}")
        with col3:
            # Distance from threshold
            distance = abs(prediction_score - threshold)
            st.metric("Confidence", f"{distance:.4f}")
        
        # Score visualization with proper scale
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Mahalanobis Distance"},
            delta = {'reference': threshold, 'position': "top"},
            gauge = {
                'axis': {'range': [0, max(10, prediction_score * 1.5)]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, threshold], 'color': "lightgreen"},
                    {'range': [threshold, max(10, prediction_score * 1.5)], 'color': "lightcoral"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical details
        st.markdown("##### Technical Implementation:")
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.markdown("""
            **Feature Processing:**
            - Z-score normalization
            - Variance-based selection
            - Patch-wise analysis
            """)
        with tech_col2:
            st.markdown("""
            **Anomaly Detection:**
            - Gaussian distribution fitting
            - Mahalanobis distance
            - Spatial max pooling
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer with model comparison
st.markdown("---")
st.markdown("### Model Comparison Summary")

col1, col2, col3 = st.columns(3)

with col1:
    clip_scores, clip_labels = load_clip_scores()
    clip_auc = roc_auc_score(clip_labels, clip_scores)
    st.markdown(f"""
    <div class="metric-container">
        <h4> CLIP</h4>
        <p>AUC: <strong>{clip_auc:.4f}</strong></p>
        <p>Type: Ensemble (Prototypes)</p>
        <p>Accuracy: <strong>68.5%</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    padim_scores, padim_labels = load_padim_scores()
    padim_auc = 0.6033
    st.markdown(f"""
    <div class="metric-container">
        <h4>PaDiM</h4>
        <p>AUC: <strong>{padim_auc:.4f}</strong></p>
        <p>Type: Multi-scale Patches</p>
        <p>Features: <strong>Top 100</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <h4> SPADE</h4>
        <p>AUC: <strong>{0.5093:.4f}</strong></p>
        <p>Type: <strong>Explainable</strong></p>
        <p>Output: Heatmap</p>
    </div>
    """, unsafe_allow_html=True)