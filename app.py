import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np

# ===================== COVER PAGE ===================== #

def Cover():
    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if os.path.exists("assets/logoits.png"):
            st.image("assets/logoits.png", width=80)
        else:
            st.write("")

    with col2:
        if os.path.exists("assets/logobme.png"):
            st.image("assets/logobme.png", width=80)
        else:
            st.write("")

    st.title("TUGAS AKHIR")
    st.header("KLASIFIKASI TINGKAT KEPARAHAN GLAUKOMA BERDASARKAN FITUR MORFOLOGI PADA CITRA FUNDUS RETINA MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK (CNN)")
    st.subheader("Nadhifatul Fuadah - 5023211053")
    st.markdown("### Dosen Pembimbing 1: Dr. Tri Arief Sardjono, S.T., M.T")
    st.markdown("### Dosen Pembimbing 2: Nada Fitrieyatul Hikmah, S.T., M.T")

    st.sidebar.info(
        "Navigation Instructions:\n"
        "- Go to **Preprocessing** to enhance image quality\n"
        "- Go to **Segmentation** to choose between OD/OC or Vessel\n"
        "- Go to **Feature Extraction** to analyze CDR, vessel tortuosity, etc.\n"
        "- Use **Classification** to predict glaucoma severity\n"
        "- Visit **About Glaucoma** to learn more"
    )

# ===================== ABOUT PAGE ===================== #

def About():
    st.title("About Glaucoma")
    st.markdown("""
    Glaucoma is a disease that damages the optic nerve due to high intraocular pressure.  
    It can lead to permanent blindness if untreated.

    **Severity Stages**:
    - Normal
    - Mild
    - Moderate
    - Severe

    **Key Morphological Indicators**:
    - Cup-to-Disc Ratio (CDR)
    - Optic disc deformation
    - Vessel tortuosity
    - Bifurcation patterns
    - Vascular narrowing
    """)

# ===================== PREPROCESSING PAGE ===================== #

def color_normalization(image, avg_r, avg_g, avg_b):
    img_float = image.astype(np.float32) / 255.0
    mean_r = np.mean(img_float[:, :, 0])
    if mean_r == 0:
        mean_r = 1e-6
    R_norm = (img_float[:, :, 0] / mean_r) * avg_r
    G_norm = (img_float[:, :, 1] / mean_r) * avg_g
    B_norm = (img_float[:, :, 2] / mean_r) * avg_b
    normalized_img = np.stack([
        np.clip(R_norm, 0, 1),
        np.clip(G_norm, 0, 1),
        np.clip(B_norm, 0, 1)], axis=2) * 255
    return normalized_img.astype(np.uint8)

def apply_gamma_correction(image, gamma=1.1):
    normalized = image / 255.0
    corrected = np.power(normalized, 1.0 / gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def apply_clahe_rgb(image, clip_limit=2.0, tile_grid_size=(12, 12)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_median_filter(image, ksize=3):
    channels = cv2.split(image)
    filtered_channels = [cv2.medianBlur(ch, ksize) for ch in channels]
    return cv2.merge(filtered_channels)

def Preprocessing():
    st.title("Preprocessing Steps")
    st.markdown("Upload a fundus image and select a specific preprocessing step from the options.")

    uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        st.image(img_np, caption="Original Image", use_column_width=True)

        step = st.radio("Select Preprocessing Step", [
            "Resize Image",
            "Color Normalization",
            "Gamma Correction",
            "CLAHE",
            "Median Filter"
        ])

        processed_img = img_np.copy()

        if step == "Resize Image":
            resized = cv2.resize(processed_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            st.image(resized, caption="Resized Image (256x256)")

        elif step == "Color Normalization":
            resized = cv2.resize(processed_img, (256, 256))
            avg_r, avg_g, avg_b = 0.5543, 0.3411, 0.1512
            norm_img = color_normalization(resized, avg_r, avg_g, avg_b)
            st.image(norm_img, caption="Color Normalization Result")

        elif step == "Gamma Correction (γ = 1.1)":
            resized = cv2.resize(processed_img, (256, 256))
            norm_img = color_normalization(resized, 0.5543, 0.3411, 0.1512)
            gamma_img = apply_gamma_correction(norm_img, gamma=1.1)
            st.image(gamma_img, caption="Gamma Correction Result")

        elif step == "CLAHE (clip=2.0, tile=12x12)":
            resized = cv2.resize(processed_img, (256, 256))
            norm_img = color_normalization(resized, 0.5543, 0.3411, 0.1512)
            gamma_img = apply_gamma_correction(norm_img, gamma=1.1)
            clahe_img = apply_clahe_rgb(gamma_img, clip_limit=2.0, tile_grid_size=(12, 12))
            st.image(clahe_img, caption="CLAHE Result")

        elif step == "Median Filter (3x3)":
            resized = cv2.resize(processed_img, (256, 256))
            norm_img = color_normalization(resized, 0.5543, 0.3411, 0.1512)
            gamma_img = apply_gamma_correction(norm_img, gamma=1.1)
            clahe_img = apply_clahe_rgb(gamma_img, clip_limit=2.0, tile_grid_size=(12, 12))
            median_img = apply_median_filter(clahe_img, ksize=3)
            st.image(median_img, caption="Median Filter Result")
            
        st.success("Preprocessing complete.")

# ===================== OTHER PAGES ===================== #

def Segmentation():
    st.title("Segmentation")
    seg_type = st.selectbox("Segmentation Type", ["Optic Disc & Cup", "Blood Vessel"])
    if seg_type == "Optic Disc & Cup":
        st.markdown("Running OD/OC segmentation model...")
    elif seg_type == "Blood Vessel":
        st.markdown("Running vessel segmentation model...")

def FeatureExtraction():
    st.title("Feature Extraction")
    feat_type = st.selectbox("Feature Source", ["OD/OC Segmentation", "Vessel Segmentation"])
    if feat_type == "OD/OC Segmentation":
        st.markdown("Extracting CDR, disc/cup area, eccentricity, solidity, etc.")
    elif feat_type == "Vessel Segmentation":
        st.markdown("Extracting vessel features: tortuosity, skeleton length, bifurcation points, etc.")

def Classification():
    st.title("Glaucoma Classification")
    st.markdown("Use the trained CNN model to classify the image into one of the glaucoma severity levels.")

def Evaluation():
    st.title("Model Evaluation")
    st.markdown("Display confusion matrix, accuracy, sensitivity, specificity, and other metrics.")

# ===================== PAGE ROUTING ===================== #

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to Page", [
    "Cover", 
    "About Glaucoma", 
    "Preprocessing", 
    "Segmentation", 
    "Feature Extraction", 
    "Classification", 
    "Evaluation"
])

if page == "Cover":
    Cover()
elif page == "About Glaucoma":
    About()
elif page == "Preprocessing":
    Preprocessing()
elif page == "Segmentation":
    Segmentation()
elif page == "Feature Extraction":
    FeatureExtraction()
elif page == "Classification":
    Classification()
elif page == "Evaluation":
    Evaluation()
