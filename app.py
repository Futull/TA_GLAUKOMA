import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from model_architecture import UNet_SE_LeakyReLU  # OD/OC
from vessel_architecture import AETUnet  # VESSEL
import torch
from torchvision import transforms

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
    st.markdown("### Dosen Pembimbing 1: Prof. Dr. Tri Arief Sardjono, S.T., M.T")
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
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        st.image(img_np, caption="🟠 Original Image", use_container_width=True)

        step = st.radio("Select preprocessing step:", [
            "Resize Image",
            "Color Normalization",
            "Gamma Correction",
            "CLAHE",
            "Median Filter"])

        resized = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_LINEAR)
        st.image(resized, caption="🔵 Resized 256x256")

        if step == "Resize Image":
            st.session_state["preprocessed_image"] = resized
            st.success("Preprocessing complete.")
            return

        avg_r, avg_g, avg_b = 0.5543, 0.3411, 0.1512
        color_norm = color_normalization(resized, avg_r, avg_g, avg_b)
        st.image(color_norm, caption="🟢 Color Normalization")

        if step == "Color Normalization":
            st.session_state["preprocessed_image"] = color_norm
            st.success("Preprocessing complete.")
            return

        gamma_img = apply_gamma_correction(color_norm, gamma=1.1)
        st.image(gamma_img, caption="🔴 Gamma Correction 1.1")

        if step == "Gamma Correction":
            st.session_state["preprocessed_image"] = gamma_img
            st.success("Preprocessing complete.")
            return

        clahe_img = apply_clahe_rgb(gamma_img, clip_limit=2.0, tile_grid_size=(12, 12))
        st.image(clahe_img, caption="🟡 CLAHE clip limit 2.0 & tile grid 12x12")

        if step == "CLAHE":
            st.session_state["preprocessed_image"] = clahe_img
            st.success("Preprocessing complete.")
            return

        median_img = apply_median_filter(clahe_img, ksize=3)
        st.image(median_img, caption="🟣 Median Filter kernel 3x3")
        st.session_state["preprocessed_image"] = median_img
        st.success("Preprocessing complete.")

# ===================== SEGMENTATION ===================== #
def Segmentation():
    st.title("Segmentation")
    image = st.session_state.get("preprocessed_image", None)
    if image is None:
        st.warning("Please complete preprocessing first.")
        return

    seg_type = st.radio("Select segmentation type:", ["Optic Disc & Cup", "Blood Vessel"])

    if st.button("🔁 Load Model & Run Segmentation"):
        with st.spinner("Processing... please wait"):
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
            img_tensor = preprocess(image).unsqueeze(0)

            if seg_type == "Optic Disc & Cup":
                model = UNet_SE_LeakyReLU(num_classes=3)
                model.load_state_dict(torch.load("models/CDR_BEST_fold_model.pt", map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    output = model(img_tensor)
                    output = torch.softmax(output, dim=1)
                    mask = output.squeeze().numpy()
                    mask = np.argmax(mask, axis=0).astype(np.uint8)
                    combined = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    combined[mask == 1] = [100, 100, 100]     # Cup - dark gray
                    combined[mask == 2] = [255, 255, 255]  # Disc - white
                    st.session_state["od_oc_mask"] = mask
                    st.session_state["cup_mask"] = (mask == 1).astype(np.uint8) * 255
                    st.session_state["disc_mask"] = (mask == 2).astype(np.uint8) * 255
                    st.session_state["od_oc_segmented"] = combined

            elif seg_type == "Blood Vessel":
                model = AETUnet()
                model.load_state_dict(torch.load("models/vessel_best_model.pth", map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    output = model(img_tensor)
                    mask = output.squeeze().numpy()
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    st.session_state["vessel_mask"] = mask
                    st.session_state["vessel_segmented"] = mask

        st.success("Segmentation completed.")

    if "od_oc_segmented" in st.session_state:
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_container_width=True)
        col2.image(st.session_state["od_oc_segmented"], caption="OD&OC Segmentation Result", use_container_width=True)

    if "vessel_segmented" in st.session_state:
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_container_width=True)
        col2.image(st.session_state["vessel_segmented"], caption="Blood Vessel Segmentation Result", clamp=True, use_container_width=True)

# ===================== OTHER PAGES ===================== #

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
    "Evaluation"])

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
