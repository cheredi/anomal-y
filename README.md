# anomal-y

## A Comparative Study using PaDiM, SPADE, and CLIP

## Overview
This project investigates anomaly detection in video surveillance by comparing three distinct approaches:

**PaDiM** – Patch distribution modeling using statistical distances

**SPADE** – Gradient-based visual attention using Grad-CAM

**CLIP** – Semantic similarity via multimodal language-image embeddings

The models are evaluated on the ShanghaiTech Campus dataset. The final deliverables include a LaTeX report, visual analysis, and a demo via a Streamlit app.

## Dataset
The project uses the ShanghaiTech Campus video dataset. To adapt for image-based models:

Frames were extracted using OpenCV

Ground truth masks were used to label normal vs abnormal images

A balanced sample of 8,000 frames was used (4,000 per class)

**Note:** The dataset is not included in the repo due to size. Please download it separately and follow the extraction instructions in /data/README.md.

# Models and Methods
**PaDiM**
Uses multivariate Gaussian modeling on ResNet features

Detects statistical outliers in patch distributions

**SPADE**
Generates Grad-CAMs for each image

Anomaly is defined as deviation from average attention map

**CLIP**
Embeds images into semantic space

Compares against prototype clusters using cosine similarity and distance metrics

# Results
Model	ROC-AUC
**PaDiM	0.603**

**SPADE	0.509**

**CLIP	0.797**

Visualizations and detailed plots are available in the figures/ folder and in the LaTeX report.

## Run the Streamlit App
To demo the project visually:
pip install -r requirements.txt
streamlit run app/app.py

# Report
The full LaTeX report is included in /report. It includes:

Dataset overview

Methodology

Model implementation

ROC curves, score distributions, and heatmaps

Comparative evaluation

Future work suggestions

# Future Directions
Add temporal modeling (I3D, Transformers, or optical flow)

Fuse statistical, visual, and semantic models into a hybrid system

Incorporate additional modalities like audio

Explore real-time deployment strategies

### Author
**Tatyana Amugo**
