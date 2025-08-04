# 🎥 Anomal-y: A Comparative Study in Video Surveillance Anomaly Detection

> **Streamlit Demo:** [anomal-y-detection.streamlit.app](https://anomal-y-detection.streamlit.app)

---

##  Overview

This project explores anomaly detection in surveillance video through **three distinct model paradigms**:

- **PaDiM** — Patch Distribution Modeling (Statistical outlier detection using multivariate Gaussians)
- **SPADE** — Gradient-based Visual Attention (Analyzing Grad-CAM activation differences)
- **CLIP** — Semantic Similarity (Language-image embeddings via vision-language models)

Using the **ShanghaiTech Campus dataset**, we extract image frames and compare the models using ROC-AUC, score distribution, and visual heatmaps. The full evaluation is available in a LaTeX report and a live interactive demo.

---

##  Dataset

The project is based on the **ShanghaiTech Campus** surveillance video dataset, which features real-world campus scenes with labeled anomalies like:

- Bicycles in pedestrian areas  
- Sudden crowding  
- Unusual object motion

**Preprocessing steps:**

- Extracted ~275,000 frames using OpenCV  
- Applied ground-truth masks to label normal vs abnormal  
- Sampled a balanced dataset of **8,000 frames** (4,000 normal, 4,000 abnormal)  

📌 **Note:** Due to dataset size, data is not included in the repository. See `data/README.md` for extraction scripts and setup.

---

## 🧪 Models & Methodologies

###  PaDiM

- Backbone: ResNet-18  
- Models spatial patch distributions using multivariate Gaussians  
- Detects statistically anomalous patches  

###  SPADE

- Computes Grad-CAM visualizations from ResNet  
- Averages normal CAMs to form a reference  
- Detects anomalies via SSIM deviation from normal attention  

###  CLIP

- Uses OpenAI’s CLIP (ViT-B/32) to embed image semantics  
- Prototypes generated via K-Means  
- Measures similarity via cosine and Euclidean metrics  

---

##  Results

| Model | ROC-AUC Score |
|-------|----------------|
| **PaDiM** | 0.603 |
| **SPADE** | 0.509 |
| **CLIP** | **0.797** |

 Visualizations (ROC curves, score distributions, and Grad-CAM heatmaps) can be found in `figures/` and the LaTeX report.

---

## 🚀 Try the Streamlit App

To launch the demo locally:

```bash
git clone https://github.com/cheredi/anomal-y.git
cd anomal-y
pip install -r requirements.txt
streamlit run app/app.py
```

Or use the hosted app: [anomal-y-detection.streamlit.app](https://anomal-y-detection.streamlit.app)

---

## 📄 Report

The full LaTeX report is available in the `/Report` folder. It includes:

- Dataset overview and extraction methodology  
- Detailed model architectures  
- Experimental results and visualizations  
- Comparative insights and limitations  
- Future directions  

---

##  Future Work

- Incorporate **temporal modeling** (e.g., I3D, Transformers, Optical Flow)  
- Use **multimodal fusion** (visual + semantic + motion)  
- Integrate **segmentation and audio cues**  
- Explore **real-time deployment** on edge devices  

---

##  Author

**Tatyana Amugo**  
Capstone Project, Department of Computer Science  
University of [Insert Name]
