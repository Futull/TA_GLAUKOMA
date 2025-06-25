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
        "- Go to *Segmentation* to choose between OD/OC or Vessel\n"
        "- Go to *Feature Extraction* to analyze CDR, vessel tortuosity, etc.\n"
        "- Use *Classification* to predict glaucoma severity\n"
    )

# ===================== PREPROCESSING FUNCTIONS ===================== #
# ===================== PREPOS OD/OC FUNCTIONS ===================== #
def crop_optic_disc_improved(image, crop_factor=1.0):
    """
    Crop around the ONH with proper margins using grayscale and thresholding method.
    Output is in RGB format for Streamlit visualization.
    """
    try:
        # Convert the image to grayscale (instead of LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding to extract the optic disc
        _, binary_image = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        # Convert binary image to RGB (3 channels)
        binary_rgb_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Select the largest contour
            optic_disc_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask for the optic disc
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [optic_disc_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            
            # Apply the mask to extract the optic disc region
            optic_disc = cv2.bitwise_and(image, image, mask=mask)
            
            # Calculate the bounding rectangle for the optic disc contour
            x, y, w, h = cv2.boundingRect(optic_disc_contour)
            
            # Calculate the center of the bounding rectangle
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate the radius of the bounding rectangle
            radius = max(w // 2, h // 2)
        else:
            # Fallback if no contour found
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            radius = min(image.shape[:2]) // 10
        
        # Calculate crop with margins
        crop_radius = int(radius * crop_factor)
        
        # Ensure bounds are within image size
        x_start = max(0, center_x - crop_radius)
        y_start = max(0, center_y - crop_radius)
        x_end = min(image.shape[1], center_x + crop_radius)
        y_end = min(image.shape[0], center_y + crop_radius)
        
        # Crop the image
        cropped_image = image[y_start:y_end, x_start:x_end]
        
        # Make it square if needed
        h, w = cropped_image.shape[:2]
        if h != w:
            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            cropped_image = cropped_image[start_h:start_h+size, start_w:start_w+size]
        
        # Return the cropped image and details in RGB format
        return cropped_image, (center_x, center_y, radius), binary_rgb_image
    
    except Exception as e:
        st.warning(f"ONH detection failed: {e}. Using fallback method.")
        
        # Fallback method
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        center_x, center_y = max_loc
        radius = min(image.shape[:2]) // 10
        crop_radius = int(radius * crop_factor)
        
        x_start = max(0, center_x - crop_radius)
        y_start = max(0, center_y - crop_radius)
        x_end = min(image.shape[1], center_x + crop_radius)
        y_end = min(image.shape[0], center_y + crop_radius)
        
        cropped_image = image[y_start:y_end, x_start:x_end]
        
        # Make it square
        h, w = cropped_image.shape[:2]
        if h != w:
            size = min(h, w)
            start_h = (h - size) // 2
            start_w = (w - size) // 2
            cropped_image = cropped_image[start_h:start_h+size, start_w:start_w+size]
        
        # Convert to RGB (even fallback images will be in RGB format)
        binary_rgb_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        
        return cropped_image, (center_x, center_y, radius), binary_rgb_image


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
    
    # Step 1: Crop ONH region
    try:
        cropped_image, detection_info, binary_rgb_image = crop_optic_disc_improved(image, crop_factor=1.0)
        
        # Menyimpan hasil pemotongan, informasi deteksi, dan mask biner RGB
        results['step1_cropped'] = cropped_image
        results['detection_info'] = detection_info
        results['binary_rgb_mask'] = binary_rgb_image
        
    except Exception as e:
        st.error(f"Error in Step 1 (ONH Cropping): {e}")
        return None
    
    # Step 2: Resize to 256x256
    resized_image = resize_image(cropped_image, target_size=(256, 256))
    results['step2_resized'] = resized_image
    
    # Step 3: Sharpening
    sharpened_image = combined_sharpening(resized_image)
    results['step3_sharpened'] = sharpened_image
    
    # Step 4: Color Normalization
    color_normalized_image = color_normalization_fixed(sharpened_image)
    results['step4_color_norm'] = color_normalized_image
    
    # Step 5: Gamma Correction
    gamma_corrected_image = apply_gamma_correction(color_normalized_image, gamma=1.1)
    results['step5_gamma'] = gamma_corrected_image
    
    # Step 6: CLAHE
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=2.0, tile_grid_size=(12, 12))
    results['step6_clahe'] = clahe_image
    
    # Step 7: Median Filter
    final_image = apply_median_filter(clahe_image, ksize=3)
    results['step7_final'] = final_image
    
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
    
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        
        st.subheader("Original Image")
        st.image(img_np, caption="Original Fundus Image", use_container_width=True)
        
        task = st.radio("Select Preprocessing Task", ["OD/OC Segmentation", "Vessel Segmentation"])
        
        if task == "OD/OC Segmentation":
            if st.button("Apply OD/OC Preprocessing"):
                with st.spinner("Processing..."):
                    results = preprocess_od_oc_stepwise(img_np)
                    
                    if results:
                        st.session_state['preprocessing_results'] = results
                        st.session_state['preprocessed_image'] = results['step7_final']
                        
                        # Display all steps in order
                        st.subheader("Preprocessing Pipeline Results")
                        
                        # Show 8 steps in 4 columns x 2 rows
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.image(img_np, caption="Original")
                        
                        with col2:
                            st.image(results['step1_cropped'], caption="1.ONH Crop")
                        
                        with col3:
                            st.image(results['step2_resized'], caption="2.Resized 256 x 256")
                        
                        with col4:
                            st.image(results['step3_sharpened'], caption="3.Sharpening")
                        
                        # Second row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.image(results['step4_color_norm'], caption="4.RGB Color Normalization")
                        
                        with col2:
                            st.image(results['step5_gamma'], caption="5.Gamma Correct")
                        
                        with col3:
                            st.image(results['step6_clahe'], caption="6.CLAHE")
                        
                        with col4:
                            st.image(results['step7_final'], caption="7.Median Filter")
                        
                        st.success("✅ OD/OC preprocessing completed!")
                        
                        # Show detection info
                        center_x, center_y, radius = results['detection_info']
                        st.info(f"ONH detected at center ({center_x}, {center_y}) with radius {radius} pixels")
                        
        elif task == "Vessel Segmentation":
            if st.button("Apply Vessel Preprocessing"):
                with st.spinner("Processing vessel segmentation..."):
                    results = preprocess_vessel_stepwise(img_np)
                    
                    st.session_state['vessel_preprocessed'] = results['step5_final']
                    st.session_state['vessel_results'] = results
                    
                    # Display all steps in order
                    st.subheader("Vessel Preprocessing Pipeline Results")
                    
                    # Show 6 steps in 3 columns x 2 rows
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(img_np, caption="Original Image")
                    
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
        st.warning("⚠ Please upload an image to begin preprocessing.")
        
# ===================== SEGMENTATION ===================== #
@st.cache_resource
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

@st.cache_resource
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
        selected_task = st.selectbox("Select Segmentation Task", available_tasks)
    
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
                    
                    # Convert prediction to displayable format
                    if len(prediction.shape) == 3:
                        od_mask = (prediction[0] > 0.5).astype(np.uint8) * 255
                        oc_mask = (prediction[1] > 0.5).astype(np.uint8) * 255
                        
                        with col2:
                            st.image(od_mask, caption="OD Segmentation")
                        
                        col3, col4 = st.columns(2)
                        with col3:
                            st.image(oc_mask, caption="OC Segmentation")
                        
                        # Save results
                        st.session_state['od_mask'] = od_mask
                        st.session_state['oc_mask'] = oc_mask
                        st.session_state['segmentation_completed'] = True
                        st.session_state['segmentation_type'] = 'OD/OC'
                        
                        st.success("✅ OD/OC segmentation completed!")
                    else:
                        mask = (prediction > 0.5).astype(np.uint8) * 255
                        with col2:
                            st.image(mask, caption="Segmentation Result")
                        st.session_state['seg_mask'] = mask
                        st.session_state['segmentation_completed'] = True
                        st.session_state['segmentation_type'] = 'OD/OC'
                        st.success("✅ Segmentation completed!")
    
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

# ===================== OTHER PAGES ===================== #

def FeatureExtraction():
    st.title("Feature Extraction")
    feat_type = st.selectbox("Feature Source", ["OD/OC Segmentation", "Vessel Segmentation"])
    
    if feat_type == "OD/OC Segmentation":
        st.markdown("""
        *OD/OC Features:* 
        - Cup-to-Disc Ratio (CDR)
        - Disc area and cup area
        - Rim area
        - Eccentricity
        - Solidity
        - Aspect ratio
        """)
        
    elif feat_type == "Vessel Segmentation":
        st.markdown("""
        *Vessel Features:*
        - Vessel tortuosity
        - Skeleton length
        - Bifurcation points
        - Vessel density
        - Average vessel width
        - Fractal dimension
        """)

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
        "Preprocessing", 
        "Segmentation", 
        "Feature Extraction", 
        "Classification", 
    ])
    
    # Route to appropriate page
    if page == "Cover":
        Cover()
    elif page == "Preprocessing":
        Preprocessing()
    elif page == "Segmentation":
        Segmentation()
    elif page == "Feature Extraction":
        FeatureExtraction()
    elif page == "Classification":
        Classification()

if __name__ == "__main__":
    main()
