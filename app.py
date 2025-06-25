import streamlit as st
from PIL import Image
import os
from skimage.measure import label, regionprops
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

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

# ===================== PREPROCESSING FUNCTIONS ===================== #

def crop_optic_disc(image):
    """
    Crop the optic nerve head (ONH) region from the fundus image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding to extract the optic disc
    _, binary_image = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        raise ValueError("No optic disc detected in the image.")
    
    # Filter contours based on area to get the optic disc contour
    optic_disc_contour = max(contours, key=cv2.contourArea)
    
    # Find the bounding rectangle for the optic disc contour
    x, y, w, h = cv2.boundingRect(optic_disc_contour)
    
    # Calculate the center of the bounding rectangle
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate the radius of the bounding rectangle
    radius = max(w // 2, h // 2)
    
    # Calculate the coordinates for the zoomed region
    zoom_x = max(0, center_x - 2 * radius)
    zoom_y = max(0, center_y - 2 * radius)
    zoom_w = min(4 * radius, image.shape[1] - zoom_x)
    zoom_h = min(4 * radius, image.shape[0] - zoom_y)
    
    # Crop and zoom the region
    cropped_zoomed_image = image[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
    
    return cropped_zoomed_image

def resize_image(image, target_size=(256, 256)):
    """
    Resize image to target size
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def unsharp_mask(image, blur_ksize=5, strength=1.0):
    """
    Apply unsharp masking for sharpening
    """
    blur = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    mask = cv2.addWeighted(image, 1 + strength, blur, -strength, 0)
    return np.clip(mask, 0, 255).astype(np.uint8)

def high_pass_filter(image):
    """
    Apply high-pass filter for additional sharpening
    """
    low_pass = cv2.GaussianBlur(image, (9, 9), 0)
    high_pass = cv2.subtract(image, low_pass)
    sharpened = cv2.add(image, high_pass)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def combined_sharpening(image):
    """
    Combine unsharp masking and high-pass filtering
    """
    unsharp = unsharp_mask(image, blur_ksize=5, strength=1.5)
    highpass = high_pass_filter(unsharp)
    return highpass

def color_normalization_fixed(image, avg_r=0.9601, avg_g=0.6374, avg_b=0.3408):
    """
    Apply per-channel color normalization
    """
    img = image.astype(np.float32) / 255.0  # normalize to [0,1]
    
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])
    
    img[:, :, 0] *= (avg_r / (mean_r + 1e-6))
    img[:, :, 1] *= (avg_g / (mean_g + 1e-6))
    img[:, :, 2] *= (avg_b / (mean_b + 1e-6))
    
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def apply_gamma_correction(image, gamma=1.1):
    """
    Apply gamma correction
    """
    if len(image.shape) == 3:  # Color image
        normalized = image / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    else:  # Grayscale image
        normalized = image / 255.0
        corrected = np.power(normalized, 1.0 / gamma)
        return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(12, 12)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    if len(image.shape) == 3:  # Color image
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:  # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def apply_median_filter(image, ksize=3):
    """
    Apply median filter for noise reduction
    """
    return cv2.medianBlur(image, ksize)

def preprocess_od_oc_stepwise(image):
    """
    Apply step-by-step preprocessing for OD/OC segmentation
    Returns a dictionary with all intermediate results
    """
    results = {}
    
    # Step 1: Crop ONH region
    try:
        cropped_image = crop_optic_disc(image)
        results['step1_cropped'] = cropped_image
    except Exception as e:
        st.error(f"Error in Step 1 (Cropping): {e}")
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

def preprocess_vessel(image):
    """
    Preprocessing for vessel segmentation
    """
    # Resize image (256x256)
    resized_image = resize_image(image, target_size=(256, 256))
    
    # Extract Green Channel
    if len(resized_image.shape) == 3:
        green_channel = resized_image[:, :, 1]  # Extract green channel
    else:
        green_channel = resized_image
    
    # Convert to 3-channel for processing
    green_3ch = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2RGB)
    
    # Gamma Correction on Green Channel
    gamma_corrected_image = apply_gamma_correction(green_3ch, gamma=1.1)
    
    # CLAHE on Green Channel
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=2.0, tile_grid_size=(12, 12))
    
    # Median Filter
    final_image = apply_median_filter(clahe_image, ksize=3)
    
    return final_image

# ===================== PREPROCESSING PAGE ===================== #

def Preprocessing():
    st.title("Preprocessing Steps")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Load and display original image
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        
        st.subheader("Original Image")
        st.image(img_np, caption="🟠 Original Fundus Image", use_container_width=True)
        
        # Task selection
        task = st.radio("Select Preprocessing Task", ["OD/OC Segmentation", "Vessel Segmentation"])
        
        if task == "OD/OC Segmentation":
            st.subheader("OD/OC Preprocessing Pipeline")
            
            # Add processing button
            if st.button("🔄 Start OD/OC Preprocessing"):
                with st.spinner("Processing... Please wait"):
                    # Process the image step by step
                    results = preprocess_od_oc_stepwise(img_np)
                    
                    if results:
                        # Store results in session state
                        st.session_state['preprocessing_results'] = results
                        st.session_state['preprocessed_image'] = results['step7_final']
                        
                        # Display all steps
                        st.subheader("Preprocessing Results")
                        
                        # Create columns for better layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(results['step1_cropped'], caption="Step 1: ONH Cropped", use_container_width=True)
                            st.image(results['step3_sharpened'], caption="Step 3: Sharpened", use_container_width=True)
                            st.image(results['step5_gamma'], caption="Step 5: Gamma Corrected", use_container_width=True)
                            st.image(results['step7_final'], caption="Step 7: Final Result", use_container_width=True)
                        
                        with col2:
                            st.image(results['step2_resized'], caption="Step 2: Resized (256x256)", use_container_width=True)
                            st.image(results['step4_color_norm'], caption="Step 4: Color Normalized", use_container_width=True)
                            st.image(results['step6_clahe'], caption="Step 6: CLAHE Applied", use_container_width=True)
                        
                        st.success("✅ Preprocessing completed successfully!")
                        
                        # Show processing summary
                        st.info("""
                        **Processing Summary:**
                        1. **ONH Cropping**: Detected and cropped optic nerve head region
                        2. **Resizing**: Resized to 256×256 pixels
                        3. **Sharpening**: Applied unsharp masking + high-pass filter
                        4. **Color Normalization**: Normalized RGB channels
                        5. **Gamma Correction**: Applied gamma correction (γ=1.1)
                        6. **CLAHE**: Applied contrast enhancement (clip=2.0, tile=12×12)
                        7. **Median Filter**: Applied noise reduction (kernel=3×3)
                        """)
                    
        elif task == "Vessel Segmentation":
            st.subheader("Vessel Preprocessing Pipeline")
            
            if st.button("🔄 Start Vessel Preprocessing"):
                with st.spinner("Processing vessel segmentation preprocessing..."):
                    processed_vessel = preprocess_vessel(img_np)
                    
                    # Store in session state
                    st.session_state['vessel_preprocessed'] = processed_vessel
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_np, caption="Original Image", use_container_width=True)
                    with col2:
                        st.image(processed_vessel, caption="Vessel Preprocessed", use_container_width=True)
                    
                    st.success("✅ Vessel preprocessing completed!")
                    
                    st.info("""
                    **Vessel Processing Summary:**
                    1. **Resizing**: Resized to 256×256 pixels
                    2. **Green Channel**: Extracted green channel
                    3. **Gamma Correction**: Applied gamma correction (γ=1.1)
                    4. **CLAHE**: Applied contrast enhancement
                    5. **Median Filter**: Applied noise reduction
                    """)
        
        # Show individual step controls (optional)
        if st.checkbox("Show Individual Step Controls"):
            st.subheader("Individual Step Processing")
            step_option = st.selectbox("Select Step", [
                "Step 1: ONH Cropping",
                "Step 2: Resize to 256x256",
                "Step 3: Sharpening",
                "Step 4: Color Normalization",
                "Step 5: Gamma Correction",
                "Step 6: CLAHE",
                "Step 7: Median Filter"
            ])
            
            if st.button(f"Apply {step_option}"):
                # Individual step processing logic can be added here
                st.info(f"Processing: {step_option}")
                
    else:
        st.warning("⚠️ Please upload an image to begin preprocessing.")
        st.info("""
        **Supported formats:** PNG, JPG, JPEG
        
        **Recommended image characteristics:**
        - Fundus retinal images
        - Clear optic disc visibility
        - Good contrast and brightness
        """)

# ===================== SEGMENTATION ===================== #
def Segmentation():
    st.title("Segmentation")
    
    # Check if preprocessing is completed
    if 'preprocessed_image' not in st.session_state and 'vessel_preprocessed' not in st.session_state:
        st.warning("⚠️ Please complete preprocessing first.")
        return
    
    seg_type = st.radio("Select segmentation type:", ["Optic Disc & Cup", "Blood Vessel"])
    
    if st.button("🔁 Load Model & Run Segmentation"):
        st.info("🔄 Loading model and running segmentation...")
        st.warning("⚠️ Model loading functionality needs to be implemented with actual model files.")
        
        # Placeholder for model loading and segmentation
        with st.spinner("Processing... please wait"):
            if seg_type == "Optic Disc & Cup":
                if 'preprocessed_image' in st.session_state:
                    image = st.session_state['preprocessed_image']
                    st.image(image, caption="Preprocessed Image for OD/OC Segmentation", use_container_width=True)
                    # Add actual segmentation logic here
                    
            elif seg_type == "Blood Vessel":
                if 'vessel_preprocessed' in st.session_state:
                    image = st.session_state['vessel_preprocessed']
                    st.image(image, caption="Preprocessed Image for Vessel Segmentation", use_container_width=True)
                    # Add actual segmentation logic here
        
        st.success("✅ Segmentation completed.")

# ===================== OTHER PAGES ===================== #

def FeatureExtraction():
    st.title("Feature Extraction")
    feat_type = st.selectbox("Feature Source", ["OD/OC Segmentation", "Vessel Segmentation"])
    
    if feat_type == "OD/OC Segmentation":
        st.markdown("""
        **OD/OC Features to Extract:**
        - Cup-to-Disc Ratio (CDR)
        - Disc area and cup area
        - Rim area
        - Eccentricity
        - Solidity
        - Aspect ratio
        """)
        
    elif feat_type == "Vessel Segmentation":
        st.markdown("""
        **Vessel Features to Extract:**
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

def Evaluation():
    st.title("Model Evaluation")
    st.markdown("""
    **Evaluation Metrics:**
    - Confusion Matrix
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    - Precision
    - F1-Score
    - ROC Curve
    - AUC Score
    """)

# ===================== PAGE ROUTING ===================== #

def main():
    st.set_page_config(
        page_title="Glaucoma Detection System",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Go to Page", [
        "Cover", 
        "About Glaucoma", 
        "Preprocessing", 
        "Segmentation", 
        "Feature Extraction", 
        "Classification", 
        "Evaluation"
    ])
    
    # Add some styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.2rem;
        color: #ff7f0e;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Route to appropriate page
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

if __name__ == "__main__":
    main()
