import streamlit as st
from PIL import Image
import os
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening, closing
from skimage.filters import gaussian
from model_architecture import Build_UNet  # OD/OC
from vessel_architecture import Build_UNet  # VESSEL
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import ndimage

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
        st.warning("⚠️ Please upload an image to begin preprocessing.")
# ===================== SEGMENTATION ===================== #
def Segmentation():
    st.title("Segmentation")
    
    if 'preprocessed_image' not in st.session_state and 'vessel_preprocessed' not in st.session_state:
        st.warning("⚠️ Please complete preprocessing first.")
        return
    
    seg_type = st.radio("Select segmentation type:", ["Optic Disc & Cup", "Blood Vessel"])
    
    if st.button("🔁 Load Model & Run Segmentation"):
        with st.spinner("Processing..."):
            if seg_type == "Optic Disc & Cup":
                if 'preprocessed_image' in st.session_state:
                    image = st.session_state['preprocessed_image']
                    st.image(image, caption="Preprocessed for OD/OC Segmentation")

                    # Check if model exists
                    if not os.path.exists('fix_model_odoc.pt'):
                        st.error("Model file not found!")
                        return
                    
                    try:
                        model_od_oc = Build_UNet(num_classes=3)
                        model_od_oc.load_state_dict(torch.load('fix_model_odoc.pt', map_location='cpu'))
                        model_od_oc.eval()

                        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
                        with torch.no_grad():
                            output = model_od_oc(image_tensor)
                            result = output.squeeze().cpu().numpy()
                            
                            if len(result.shape) == 3:
                                result = np.argmax(result, axis=0)
                        
                        st.image(result, caption="OD/OC Segmentation", cmap='gray')
                        st.session_state['od_oc_result'] = result
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
            elif seg_type == "Blood Vessel":
                if 'vessel_preprocessed' in st.session_state:
                    image = st.session_state['vessel_preprocessed']
                    st.image(image, caption="Preprocessed for Vessel Segmentation")

                    if not os.path.exists('fix_model_vessel.pth'):
                        st.error("Model file not found!")
                        return
                    
                    try:
                        model_vessel = Build_UNet(num_classes=1)
                        model_vessel.load_state_dict(torch.load('fix_model_vessel.pth', map_location='cpu'))
                        model_vessel.eval()

                        if len(image.shape) == 3:
                            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                            image_3ch = np.stack([image_gray] * 3, axis=-1)
                        else:
                            image_3ch = np.stack([image] * 3, axis=-1)
                        
                        image_tensor = transforms.ToTensor()(image_3ch).unsqueeze(0)
                        with torch.no_grad():
                            output = model_vessel(image_tensor)
                            result = torch.sigmoid(output).squeeze().cpu().numpy()
                            result = (result > 0.5).astype(np.uint8) * 255
                        
                        st.image(result, caption="Vessel Segmentation", cmap='gray')
                        st.session_state['vessel_result'] = result
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

# ===================== OTHER PAGES ===================== #

def FeatureExtraction():
    st.title("Feature Extraction")
    feat_type = st.selectbox("Feature Source", ["OD/OC Segmentation", "Vessel Segmentation"])
    
    if feat_type == "OD/OC Segmentation":
        pass
        
    elif feat_type == "Vessel Segmentation":
        pass

def Classification():
    st.title("Glaucoma Classification")
    st.markdown("""
    **Classification Pipeline:**
    1. Load extracted features
    2. Apply trained CNN model
    3. Predict glaucoma severity level
    
    **Severity Levels:**
    - 🟢 Normal
    - 🟡 Mild Glaucoma
    - 🟠 Moderate Glaucoma
    - 🔴 Severe Glaucoma
    """)

# ===================== PAGE ROUTING ===================== #

def main():
    st.set_page_config(
        page_title="Glaucoma Detection System",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("👁️ Navigation")
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
