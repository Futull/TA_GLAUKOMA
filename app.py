import streamlit as st
from PIL import Image
import os 
from skimage.measure import label, regionprops 
from skimage.morphology import disk, opening, closing 
from skimage.filters import gaussian
import cv2
import numpy as np 
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from model_architecture import Build_UNet  # OD/OC
from vessel_architecture import Build_UNet_Vessel  # VESSEL


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
        "- Go to *Preprocessing* to enhance image quality\n"
        "- Go to *Segmentation* To detect and identify the Optic Disc (OD), Optic Cup (OC), and retinal blood vessels\n"
        "- Go to *Feature Extraction* to analyze CDR, vessel tortuosity, etc.\n"
        "- Use *Classification* to predict glaucoma severity\n"
    )

# ===================== PREPROCESSING FUNCTIONS ===================== #
# ===================== PREPOS OD/OC FUNCTIONS ===================== #

def resize_image(image, target_size=(256, 256)):
    """Resize image to target size"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def unsharp_mask(image, blur_ksize=5, strength=1.0):
    """Apply unsharp masking for sharpening"""
    blur = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    mask = cv2.addWeighted(image, 1 + strength, blur, -strength, 0)
    return np.clip(mask, 0, 255).astype(np.uint8)

def high_pass_filter(image):
    """Apply high-pass filter for additional sharpening"""
    low_pass = cv2.GaussianBlur(image, (9, 9), 0)
    high_pass = cv2.subtract(image, low_pass)
    sharpened = cv2.add(image, high_pass)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def combined_sharpening(image):
    """Combine unsharp masking and high-pass filtering"""
    unsharp = unsharp_mask(image, blur_ksize=5, strength=1.5)
    highpass = high_pass_filter(unsharp)
    return highpass

def color_normalization_fixed(image, avg_r=0.9601, avg_g=0.6374, avg_b=0.3408):
    """Apply per-channel color normalization"""
    img = image.astype(np.float32) / 255.0
    
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])
    
    img[:, :, 0] *= (avg_r / (mean_r + 1e-6))
    img[:, :, 1] *= (avg_g / (mean_g + 1e-6))
    img[:, :, 2] *= (avg_b / (mean_b + 1e-6))
    
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def apply_gamma_correction(image, gamma=1.1):
    """Apply gamma correction"""
    normalized = image / 255.0
    corrected = np.power(normalized, 1.0 / gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(12, 12)):
    """Apply CLAHE"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def apply_median_filter(image, ksize=3):
    """Apply median filter for noise reduction"""
    return cv2.medianBlur(image, ksize)
    
# ===================== PIPELINE PREPOS OD/OC STEPS ===================== #
def preprocess_od_oc_stepwise(image):
    """
    Apply step-by-step preprocessing for OD/OC segmentation
    Returns a dictionary with all intermediate results
    """
    results = {}
    
    # Step 1: Resize to 256x256
    resized_image = resize_image(image, target_size=(256, 256))
    results['step1_resized'] = resized_image
    
    # Step 2: Sharpening
    sharpened_image = combined_sharpening(resized_image)
    results['step2_sharpened'] = sharpened_image
    
    # Step 3: Color Normalization
    color_normalized_image = color_normalization_fixed(sharpened_image)
    results['step3_color_norm'] = color_normalized_image
    
    # Step 4: Gamma Correction
    gamma_corrected_image = apply_gamma_correction(color_normalized_image, gamma=1.1)
    results['step4_gamma'] = gamma_corrected_image
    
    # Step 5: CLAHE
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=2.0, tile_grid_size=(12, 12))
    results['step5_clahe'] = clahe_image
    
    # Step 6: Median Filter
    final_image = apply_median_filter(clahe_image, ksize=3)
    results['step6_final'] = final_image
    
    return results

# ===================== PIPELINE PREPOS VESSEL STEPS ===================== #
def preprocess_vessel_stepwise(image):
    """
    Apply step-by-step preprocessing for vessel segmentation
    Returns a dictionary with all intermediate results
    """
    results = {}
    
    # Step 1: Resize to 256x256
    resized_image = resize_image(image, target_size=(256, 256))
    results['step1_resized'] = resized_image
    
    # Step 2: Green channel extraction
    if len(resized_image.shape) == 3:
        green_channel = resized_image[:, :, 1]
    else:
        green_channel = resized_image
    
    # Convert to 3-channel for display
    green_3ch = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2RGB)
    results['step2_green'] = green_3ch
    
    # Step 3: Gamma correction
    gamma_corrected_image = apply_gamma_correction(green_3ch, gamma=1.1)
    results['step3_gamma'] = gamma_corrected_image
    
    # Step 4: CLAHE enhancement
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=2.0, tile_grid_size=(8, 8))
    results['step4_clahe'] = clahe_image
    
    # Step 5: Median filter
    final_image = apply_median_filter(clahe_image, ksize=3)
    results['step5_final'] = final_image
    
    return results

# ===================== PREPROCESSING PAGE ===================== #

def Preprocessing():
    st.title("Preprocessing Steps")
    
    task = st.radio("Select Preprocessing Task", ["OD/OC Segmentation", "Vessel Segmentation"])
    
    if task == "OD/OC Segmentation":
        st.subheader("OD/OC Segmentation Preprocessing")
        uploaded_file_od = st.file_uploader("Upload Cropped ONH Image for OD/OC", type=["png", "jpg", "jpeg"], key="od_oc_upload")
        
        if uploaded_file_od:
            image_od = Image.open(uploaded_file_od).convert('RGB')
            img_np_od = np.array(image_od)
            
            st.subheader("Original Cropped Image")
            st.image(img_np_od, caption="Original Cropped ONH Image", use_container_width=True)
            
            if st.button("Apply OD/OC Preprocessing"):
                with st.spinner("Processing..."):
                    results = preprocess_od_oc_stepwise(img_np_od)
                    
                    st.session_state['preprocessing_results'] = results
                    st.session_state['preprocessed_image'] = results['step6_final']
                    
                    # Display all steps in order
                    st.subheader("OD/OC Preprocessing Pipeline Results")
                    
                    # Show 7 steps in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.image(img_np_od, caption="Original Cropped")
                    
                    with col2:
                        st.image(results['step1_resized'], caption="1.Resized 256 x 256")
                    
                    with col3:
                        st.image(results['step2_sharpened'], caption="2.Sharpening")
                    
                    with col4:
                        st.image(results['step3_color_norm'], caption="3.RGB Color Normalization")
                    
                    # Second row
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(results['step4_gamma'], caption="4.Gamma Correct")
                    
                    with col2:
                        st.image(results['step5_clahe'], caption="5.CLAHE")
                    
                    with col3:
                        st.image(results['step6_final'], caption="6.Median Filter")
                    
                    st.success("✅ OD/OC preprocessing completed!")
        else:
            st.warning("⚠ Please upload a cropped ONH image for OD/OC preprocessing.")
            
    elif task == "Vessel Segmentation":
        st.subheader("Vessel Segmentation Preprocessing")
        uploaded_file_vessel = st.file_uploader("Upload Full Fundus Image for Vessel", type=["png", "jpg", "jpeg"], key="vessel_upload")
        
        if uploaded_file_vessel:
            image_vessel = Image.open(uploaded_file_vessel).convert('RGB')
            img_np_vessel = np.array(image_vessel)
            
            st.subheader("Original Full Image")
            st.image(img_np_vessel, caption="Original Full Fundus Image", use_container_width=True)
            
            if st.button("Apply Vessel Preprocessing"):
                with st.spinner("Processing vessel segmentation..."):
                    results = preprocess_vessel_stepwise(img_np_vessel)
                    
                    st.session_state['vessel_preprocessed'] = results['step5_final']
                    st.session_state['vessel_results'] = results
                    
                    # Display all steps in order
                    st.subheader("Vessel Preprocessing Pipeline Results")
                    
                    # Show 6 steps in 3 columns x 2 rows
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(img_np_vessel, caption="Original Image")
                    
                    with col2:
                        st.image(results['step1_resized'], caption="1.Resized 256 x 256")
                    
                    with col3:
                        st.image(results['step2_green'], caption="2.Green Channel")
                    
                    # Second row
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(results['step3_gamma'], caption="3.Gamma Correct")
                    
                    with col2:
                        st.image(results['step4_clahe'], caption="4.CLAHE")
                    
                    with col3:
                        st.image(results['step5_final'], caption="5.Median Filter")
                    
                    st.success("✅ Vessel preprocessing completed!")
        else:
            st.warning("⚠ Please upload a full fundus image for vessel preprocessing.")
# ===================== SEGMENTATION ===================== #
def load_od_oc_model():
    """Load OD/OC segmentation model"""
    try:
        model = Build_UNet()
        model.load_state_dict(torch.load('models/fix_model_odoc.pt', map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading OD/OC model: {e}")
        return None

def load_vessel_model():
    """Load vessel segmentation model"""
    try:
        model = Build_UNet_Vessel()
        model.load_state_dict(torch.load('models/fix_model_vessel.pt', map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading vessel model: {e}")
        return None

def predict_od_oc(model, image):
    """Predict OD/OC segmentation"""
    # Prepare image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.squeeze(0).cpu().numpy()
    
    return prediction

def predict_vessel(model, image):
    """Predict vessel segmentation"""
    # Stack green channel to 3 channels
    if len(image.shape) == 3:
        green_channel = image[:, :, 1]
    else:
        green_channel = image
    
    # Stack to 3 channels
    image_3ch = np.stack([green_channel, green_channel, green_channel], axis=-1)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image_3ch, np.ndarray):
        image_3ch = Image.fromarray(image_3ch)
    
    input_tensor = transform(image_3ch).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.squeeze(0).cpu().numpy()
    
    return prediction

# ===================== SEGMENTATION PAGE ===================== #

def Segmentation():
    st.title("Segmentation")
    
    # Check if any preprocessing has been completed
    has_od_oc = 'preprocessed_image' in st.session_state
    has_vessel = 'vessel_preprocessed' in st.session_state
    
    if not has_od_oc and not has_vessel:
        st.warning("⚠ Please complete preprocessing first.")
        return
    
    # Let user choose which segmentation to perform
    available_tasks = []
    if has_od_oc:
        available_tasks.append("OD/OC Segmentation")
    if has_vessel:
        available_tasks.append("Vessel Segmentation")
    
    if len(available_tasks) == 1:
        selected_task = available_tasks[0]
        st.info(f"Available task: {selected_task}")
    else:
        selected_task = st.selectbox("Select Segmentation", available_tasks)
    
    # Ensure correct indentation for the following line
    if selected_task == "OD/OC Segmentation":
        st.subheader("OD/OC Segmentation")
    
        preprocessed_img = st.session_state['preprocessed_image']
    
        col1, col2 = st.columns(2)
        with col1:
            st.image(preprocessed_img, caption="Preprocessed Image")
        
        if st.button("Run OD/OC Segmentation"):
            with st.spinner("Loading model and running segmentation..."):
                model = load_od_oc_model()
                
                if model is not None:
                    prediction = predict_od_oc(model, preprocessed_img)
                    
                    # Convert prediction to colored segmentation map
                    if len(prediction.shape) == 3 and prediction.shape[0] > 1:
                        # Multi-class output - get the class with highest probability
                        seg_map = np.argmax(prediction, axis=0)
                    else:
                        # Single output - assume it's already class indices
                        seg_map = prediction.squeeze()
                    
                    # Create colored visualization
                    # 0 = Background (black), 1 = OD (red), 2 = OC (green)
                    colored_result = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
                    
                    # Background stays black (0, 0, 0)
                    colored_result[seg_map == 1] = [169, 169, 169]  # OD = Grey
                    colored_result[seg_map == 2] = [255, 255, 255]  # OC = White
                    
                    with col2:
                        st.image(colored_result, caption="OD/OC Segmentation (Red: OD, Green: OC)")
                    
                    # Show legend
                    st.subheader("Legend")
                    legend_col1, legend_col2, legend_col3 = st.columns(3)
                    with legend_col1:
                        st.markdown("🩶 **Red: Optic Disc (OD)**")
                    with legend_col2:
                        st.markdown("🤍 **White: Optic Cup (OC)**")
                    with legend_col3:
                        st.markdown("🖤 **Black: Background**")
                    
                    # Save results
                    st.session_state['segmentation_map'] = seg_map
                    st.session_state['colored_segmentation'] = colored_result
                    st.session_state['segmentation_completed'] = True
                    st.session_state['segmentation_type'] = 'OD/OC'
                    
                    st.success("✅ OD/OC segmentation completed!")

    
    elif selected_task == "Vessel Segmentation":
        st.subheader("Vessel Segmentation")
        
        preprocessed_img = st.session_state['vessel_preprocessed']
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(preprocessed_img, caption="Preprocessed Image")
        
        if st.button("Run Vessel Segmentation"):
            with st.spinner("Loading model and running segmentation..."):
                model = load_vessel_model()
                
                if model is not None:
                    prediction = predict_vessel(model, preprocessed_img)
                    
                    # Convert prediction to displayable format
                    vessel_mask = (prediction > 0.5).astype(np.uint8) * 255
                    if len(vessel_mask.shape) == 3:
                        vessel_mask = vessel_mask[0]
                    
                    with col2:
                        st.image(vessel_mask, caption="Vessel Segmentation")
                    
                    # Save results
                    st.session_state['vessel_mask'] = vessel_mask
                    st.session_state['segmentation_completed'] = True
                    st.session_state['segmentation_type'] = 'Vessel'
                    
                    st.success("✅ Vessel segmentation completed!")

# ===================== EXTRACTION FEATURE PAGES ===================== #
def get_largest_region(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return np.zeros_like(mask)
    largest = max(props, key=lambda x: x.area)
    return (labeled == largest.label).astype(np.uint8)

def postprocess_mask(mask, disc_min=500, cup_min=200):
    disc = (mask == 1).astype(np.uint8)
    cup  = (mask == 2).astype(np.uint8)

    disc = remove_small_objects(disc.astype(bool), min_size=disc_min)
    disc = remove_small_holes(disc, area_threshold=200)
    disc = get_largest_region(disc.astype(np.uint8))

    cup = remove_small_objects(cup.astype(bool), min_size=cup_min)
    cup = remove_small_holes(cup, area_threshold=100)
    cup = get_largest_region(cup.astype(np.uint8))

    final = np.zeros_like(mask, dtype=np.uint8)
    final[disc == 1] = 1
    final[cup == 1] = 2
    return final, disc, cup

def extract_od_oc_features(mask, preprocessed_image):
    try:
        disc_props = regionprops(label(mask == 1))  # Disc is labeled as 1
        cup_props = regionprops(label(mask == 2))   # Cup is labeled as 2

        if not disc_props or not cup_props:
            return None

        disc = disc_props[0]
        cup = cup_props[0]

        # Perhitungan CDR (Cup-to-Disc Ratio)
        h_cup = cup.bbox[2] - cup.bbox[0]
        h_disc = disc.bbox[2] - disc.bbox[0]
        cdr_vertical = h_cup / (h_disc + 1e-8)
        cdr_area = cup.area / (disc.area + 1e-8)
        cdr_diameter = cup.major_axis_length / (disc.major_axis_length + 1e-8)

        # Extract GLCM features for OD/OC regions (use preprocessed image)
        glcm_features = extract_glcm_features(preprocessed_image)

        features = {
            "cdr_vertical": round(cdr_vertical, 4),
            "cdr_area": round(cdr_area, 4),
            "cdr_diameter": round(cdr_diameter, 4),

            "cup_area": cup.area,
            "cup_perimeter": cup.perimeter,
            "cup_eccentricity": round(cup.eccentricity, 4),
            "cup_major_axis": round(cup.major_axis_length, 2),
            "cup_minor_axis": round(cup.minor_axis_length, 2),
            "cup_extent": round(cup.extent, 4),
            "cup_solidity": round(cup.solidity, 4),
            "cup_equiv_diameter": round(cup.equivalent_diameter, 2),

            "disc_area": disc.area,
            "disc_perimeter": disc.perimeter,
            "disc_eccentricity": round(disc.eccentricity, 4),
            "disc_major_axis": round(disc.major_axis_length, 2),
            "disc_minor_axis": round(disc.minor_axis_length, 2),
            "disc_extent": round(disc.extent, 4),
            "disc_solidity": round(disc.solidity, 4),
            "disc_equiv_diameter": round(disc.equivalent_diameter, 2),

            **glcm_features  # Add GLCM features to the dictionary
        }

        return features
    except Exception as e:
        st.error(f"Error extracting OD/OC features: {e}")
        return None

def extract_glcm_features(image):
    """Extract GLCM features for OD/OC"""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        # Convert the image to grayscale and compute GLCM
        image_ubyte = img_as_ubyte(gray_image)
        glcm = graycomatrix(image_ubyte, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        
        # Extract GLCM properties
        glcm_props = ['contrast', 'correlation', 'energy', 'homogeneity']
        glcm_features = {f"{prop}_odoc": np.mean(graycoprops(glcm, prop)) for prop in glcm_props}

        return glcm_features
    except Exception as e:
        st.error(f"Error extracting GLCM features: {e}")
        return {}

def extract_vessel_features(vessel_image):
    """Extract vessel features"""
    try:
        # Convert to grayscale if needed
        if len(vessel_image.shape) == 3:
            vessel_gray = cv2.cvtColor(vessel_image, cv2.COLOR_RGB2GRAY)
        else:
            vessel_gray = vessel_image
            
        # Example: Calculating the vessel tortuosity (can be replaced with actual calculation)
        tortuosity = np.sum(vessel_gray) / vessel_gray.shape[0]  # Simplified tortuosity measure

        # More complex features like skeleton length, bifurcation points, etc., can be added here
        features = {
            "vessel_tortuosity": round(tortuosity, 4),
            "vessel_density": np.sum(vessel_gray) / (vessel_gray.shape[0] * vessel_gray.shape[1]),
            "vessel_skeleton_length": np.sum(vessel_gray),  # Simplified feature (just an example)
        }

        return features
    except Exception as e:
        st.error(f"Error extracting vessel features: {e}")
        return None

def FeatureExtraction():
    st.title("Feature Extraction")

    # Use radio buttons to choose the feature extraction mode
    extraction_mode = st.radio("Select Segmentation for Feature Extraction", ("OD/OC Feature Extraction", "Vessel Feature Extraction"))

    if extraction_mode == "OD/OC Feature Extraction":
        st.subheader("OD/OC Feature Extraction")

        segmentation_result = st.session_state.get('segmentation_map', None)
        preprocessed_image = st.session_state.get('preprocessed_image', None)

        if segmentation_result is not None and preprocessed_image is not None:
            # Display the colored segmentation for visualization
            colored_seg = st.session_state.get('colored_segmentation', None)
            if colored_seg is not None:
                st.image(colored_seg, caption="Segmented OD/OC Image", use_container_width=True)

            # Extract OD/OC features using the segmentation map
            features = extract_od_oc_features(segmentation_result, preprocessed_image)
            if features:
                # Store features in session_state to preserve them
                st.session_state['od_oc_features'] = features

                # Display features in DataFrame
                od_oc_df = pd.DataFrame([features])
                st.subheader("OD/OC Extracted Features")
                st.dataframe(od_oc_df)

                # Plotting the extracted OD/OC features
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(features.keys(), features.values())

def Classification():
    st.title("Glaucoma Classification")
    st.markdown("""
    *Classification Pipeline:*
    1. Load extracted features
    2. Apply trained CNN model
    3. Predict glaucoma severity level
    
    *Severity Levels:*
    - 🟢 Normal
    - 🟡 Mild Glaucoma
    - 🟠 Moderate Glaucoma
    - 🔴 Severe Glaucoma
    """)

# ===================== PAGE ROUTING ===================== #

def main():
    st.set_page_config(
        page_title="Glaucoma Detection System",
        page_icon="👁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("👁 Navigation")
    page = st.sidebar.selectbox("Go to Page", [
        "Cover", 
        "1.Preprocessing", 
        "2.Segmentation", 
        "3.Feature Extraction", 
        "4.Classification", 
    ])
    
    # Route to appropriate page
    if page == "Cover":
        Cover()
    elif page == "1.Preprocessing":
        Preprocessing()
    elif page == "2.Segmentation":
        Segmentation()
    elif page == "3.Feature Extraction":
        FeatureExtraction()
    elif page == "4.Classification":
        Classification()

if __name__ == "__main__":
    main()
