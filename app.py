import streamlit as st
from PIL import Image
import os
from skimage.measure import label, regionprops
import cv2
import numpy as np
from model_architecture import Build_UNet  # OD/OC
from vessel_architecture import Build_UNet # VESSEL
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

def detect_and_crop_od(image, margin_ratio=1.0):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("Tidak ada disc optik yang terdeteksi.")
    optic_disc_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(optic_disc_contour)
    margin = int(max(w, h) * margin_ratio)
    zoom_x = max(0, x - margin)
    zoom_y = max(0, y - margin)
    zoom_w = min(w + 2 * margin, image.shape[1] - zoom_x)
    zoom_h = min(h + 2 * margin, image.shape[0] - zoom_y)
    cropped_image = image[zoom_y:zoom_y + zoom_h, zoom_x:zoom_x + zoom_w]
    return cropped_image

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def unsharp_mask(image, blur_ksize=5, strength=1.0):
    blur = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    mask = cv2.addWeighted(image, 1 + strength, blur, -strength, 0)
    return np.clip(mask, 0, 255).astype(np.uint8)

def apply_gamma_correction(image, gamma=1.1):
    normalized = image / 255.0
    corrected = np.power(normalized, 1.0 / gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(12, 12)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def color_normalization(image, avg_r, avg_g, avg_b):
    img = image.astype(np.float32) / 255.0
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])
    img[:, :, 0] *= (avg_r / (mean_r + 1e-6))
    img[:, :, 1] *= (avg_g / (mean_g + 1e-6))
    img[:, :, 2] *= (avg_b / (mean_b + 1e-6))
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def preprocess_od_oc(image, margin_ratio=1.0):
    # Step 1: Crop around ONH
    cropped_image = detect_and_crop_od(image, margin_ratio)
    st.image(cropped_image, caption="1. Cropped Image", use_container_width=True)

    # Step 2: Resize image
    resized_image = resize_image(cropped_image, target_size=(256, 256))
    st.image(resized_image, caption="2. Resized Image", use_container_width=True)

    # Step 3: Sharpening
    sharpened_image = unsharp_mask(resized_image, blur_ksize=5, strength=1.5)
    st.image(sharpened_image, caption="3. Sharpened Image", use_container_width=True)

    # Step 4: Color Normalization
    avg_r, avg_g, avg_b = 0.9601, 0.6374, 0.3408
    color_normalized_image = color_normalization(sharpened_image, avg_r, avg_g, avg_b)
    st.image(color_normalized_image, caption="4. Color Normalized Image", use_container_width=True)

    # Step 5: Gamma Correction
    gamma_corrected_image = apply_gamma_correction(color_normalized_image, gamma=1.1)
    st.image(gamma_corrected_image, caption="5. Gamma Corrected Image", use_container_width=True)

    # Step 6: CLAHE
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=2.0, tile_grid_size=(12, 12))
    st.image(clahe_image, caption="6. CLAHE Image", use_container_width=True)

    # Step 7: Median Filter
    final_image = apply_median_filter(clahe_image, ksize=3)
    st.image(final_image, caption="7. Final Processed Image", use_container_width=True)

    return final_image

# ===================== Fungsi Preprocessing untuk Vessel Segmentation ===================== #

# Fungsi Preprocessing untuk Vessel Segmentation
def preprocess_vessel(image, margin_ratio=1.0):
    # Resize image (256x256)
    resized_image = resize_image(image, target_size=(256, 256))

    # Extract Green Channel
    green_channel = resized_image[:, :, 1]  # Ambil channel hijau (Green)

    # Gamma Correction pada Green Channel
    gamma_corrected_image = apply_gamma_correction(green_channel, gamma=1.1)

    # CLAHE pada Green Channel
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=2.0, tile_grid_size=(12, 12))

    # Median Filter
    final_image = apply_median_filter(clahe_image, ksize=3)

    return final_image

# ===================== Fungsi Streamlit ===================== #


def Preprocessing():
    st.title("Preprocessing Steps")
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        st.image(img_np, caption="🟠 Original Image", use_container_width=True)

        # Select preprocessing task
        task = st.radio("Select Preprocessing Task", ["OD/OC Segmentation", "Vessel Segmentation"])

        # Process OD/OC Segmentation
        if task == "OD/OC Segmentation":
            if st.button("Apply Preprocessing for OD/OC Segmentation"):
                try:
                    processed_image = preprocess_od_oc(img_np, margin_ratio=1.0)
                    st.image(processed_image, caption="🟣 Processed Image (OD/OC Cropped)")
                    st.success("Preprocessing complete.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Process Vessel Segmentation (You can add this logic if needed)
        elif task == "Vessel Segmentation":
            pass

    else:
        st.warning("Please upload an image to begin preprocessing.")

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
