import streamlit as st
from PIL import Image
import cv2
import numpy as np
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
    st.header("KLASIFIKASI TINGKAT KEPARAHAN GLAUKOMA BERDASARKAN FITUR MORFOLOGI PADA CITRA FUNDUS RETINA MENGGUNAKAN CNN")
    st.subheader("Nadhifatul Fuadah - 5023211053")
    st.write("**Dosen Pembimbing 1:** Prof. Dr. Tri Arief Sardjono, S.T., M.T")
    st.write("**Dosen Pembimbing 2:** Nada Fitrieyatul Hikmah, S.T., M.T")

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
    st.title("Tentang Glaukoma")
    st.write("""
    Glaukoma adalah penyakit yang merusak saraf optik akibat tekanan intraokular tinggi.
    Dapat menyebabkan kebutaan permanen jika tidak diobati.
    """)
    
    st.subheader("Tingkat Keparahan:")
    st.write("- Normal")
    st.write("- Mild")  
    st.write("- Moderate")
    st.write("- Severe")
    
    st.subheader("Indikator Morfologi Utama:")
    st.write("- Cup-to-Disc Ratio (CDR)")
    st.write("- Deformasi optic disc")
    st.write("- Vessel tortuosity")
    st.write("- Pola bifurkasi")

# ===================== PREPROCESSING FUNCTIONS ===================== #
def detect_optic_disc(image):
    """Deteksi optic disc sederhana"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Cari area paling terang
    threshold_value = np.percentile(blurred, 95)
    _, bright_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    else:
        # Fallback ke center
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 8
        angles = np.linspace(0, 2*np.pi, 100)
        contour_points = np.array([[int(center_x + radius * np.cos(angle)), 
                                   int(center_y + radius * np.sin(angle))] for angle in angles])
        return contour_points.reshape(-1, 1, 2).astype(np.int32)

def crop_optic_disc(image):
    """Crop optic disc region"""
    try:
        contour = detect_optic_disc(image)
        x, y, w, h = cv2.boundingRect(contour)
        
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2
        crop_radius = int(radius * 2.5)
        
        x_start = max(0, center_x - crop_radius)
        y_start = max(0, center_y - crop_radius)
        x_end = min(image.shape[1], center_x + crop_radius)
        y_end = min(image.shape[0], center_y + crop_radius)
        
        cropped = image[y_start:y_end, x_start:x_end]
        
        # Pastikan ukuran minimum
        if cropped.shape[0] < 100 or cropped.shape[1] < 100:
            h, w = image.shape[:2]
            size = min(h, w) // 3
            center_x, center_y = w // 2, h // 2
            x_start = max(0, center_x - size // 2)
            y_start = max(0, center_y - size // 2)
            x_end = min(w, center_x + size // 2)
            y_end = min(h, center_y + size // 2)
            cropped = image[y_start:y_end, x_start:x_end]
        
        return cropped
    except:
        # Fallback ke center crop
        h, w = image.shape[:2]
        size = min(h, w) // 3
        center_x, center_y = w // 2, h // 2
        x_start = max(0, center_x - size // 2)
        y_start = max(0, center_y - size // 2)
        x_end = min(w, center_x + size // 2)
        y_end = min(h, center_y + size // 2)
        return image[y_start:y_end, x_start:x_end]

def resize_image(image, size=(256, 256)):
    """Resize image"""
    return cv2.resize(image, size)

def apply_clahe(image):
    """Apply CLAHE"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

def preprocess_od_oc(image):
    """Preprocessing untuk OD/OC segmentation"""
    # Step 1: Crop ONH
    cropped = crop_optic_disc(image)
    
    # Step 2: Resize
    resized = resize_image(cropped, (256, 256))
    
    # Step 3: CLAHE
    enhanced = apply_clahe(resized)
    
    # Step 4: Median filter
    final = cv2.medianBlur(enhanced, 3)
    
    return {
        'cropped': cropped,
        'resized': resized, 
        'enhanced': enhanced,
        'final': final
    }

def preprocess_vessel(image):
    """Preprocessing untuk vessel segmentation"""
    # Resize
    resized = resize_image(image, (256, 256))
    
    # Extract green channel
    if len(resized.shape) == 3:
        green = resized[:, :, 1]
        green_3ch = cv2.cvtColor(green, cv2.COLOR_GRAY2RGB)
    else:
        green_3ch = resized
    
    # CLAHE
    enhanced = apply_clahe(green_3ch)
    
    # Median filter
    final = cv2.medianBlur(enhanced, 3)
    
    return final

# ===================== PREPROCESSING PAGE ===================== #
def Preprocessing():
    st.title("Preprocessing")
    
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(image)
        
        st.subheader("Original Image")
        st.image(img_np, caption="Original Fundus Image")
        
        task = st.radio("Select Task", ["OD/OC Segmentation", "Vessel Segmentation"])
        
        if st.button("Start Preprocessing"):
            with st.spinner("Processing..."):
                if task == "OD/OC Segmentation":
                    results = preprocess_od_oc(img_np)
                    st.session_state['preprocessed_image'] = results['final']
                    
                    st.subheader("Preprocessing Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(results['cropped'], caption="1. Cropped ONH")
                        st.image(results['enhanced'], caption="3. CLAHE Applied")
                    
                    with col2:
                        st.image(results['resized'], caption="2. Resized 256x256")
                        st.image(results['final'], caption="4. Final Result")
                    
                elif task == "Vessel Segmentation":
                    result = preprocess_vessel(img_np)
                    st.session_state['vessel_preprocessed'] = result
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_np, caption="Original")
                    with col2:
                        st.image(result, caption="Vessel Preprocessed")
                
                st.success("Preprocessing completed!")
    else:
        st.warning("Please upload an image first.")

# ===================== SEGMENTATION ===================== #
def Segmentation():
    st.title("Segmentation")
    
    if 'preprocessed_image' not in st.session_state and 'vessel_preprocessed' not in st.session_state:
        st.warning("Please complete preprocessing first.")
        return
    
    seg_type = st.radio("Select segmentation type:", ["Optic Disc & Cup", "Blood Vessel"])
    
    if st.button("Run Segmentation"):
        st.info("Model loading functionality needs to be implemented.")
        
        if seg_type == "Optic Disc & Cup" and 'preprocessed_image' in st.session_state:
            st.image(st.session_state['preprocessed_image'], caption="Ready for OD/OC Segmentation")
        elif seg_type == "Blood Vessel" and 'vessel_preprocessed' in st.session_state:
            st.image(st.session_state['vessel_preprocessed'], caption="Ready for Vessel Segmentation")

# ===================== FEATURE EXTRACTION ===================== #
def FeatureExtraction():
    st.title("Feature Extraction")
    
    feat_type = st.selectbox("Feature Source", ["OD/OC Segmentation", "Vessel Segmentation"])
    
    if feat_type == "OD/OC Segmentation":
        st.subheader("OD/OC Features:")
        st.write("- Cup-to-Disc Ratio (CDR)")
        st.write("- Disc area and cup area") 
        st.write("- Rim area")
        st.write("- Eccentricity")
        
    elif feat_type == "Vessel Segmentation":
        st.subheader("Vessel Features:")
        st.write("- Vessel tortuosity")
        st.write("- Skeleton length")
        st.write("- Bifurcation points")
        st.write("- Vessel density")

# ===================== CLASSIFICATION ===================== #
def Classification():
    st.title("Glaucoma Classification")
    st.write("Classification model will be implemented here.")

# ===================== EVALUATION ===================== #
def Evaluation():
    st.title("Model Evaluation")
    st.write("Model evaluation metrics will be shown here.")

# ===================== MAIN FUNCTION ===================== #
def main():
    st.set_page_config(
        page_title="Glaucoma Detection System",
        page_icon="👁️",
        layout="wide"
    )
    
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
    
    # Route pages
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
