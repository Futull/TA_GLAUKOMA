import streamlit as st
from PIL import Image
import os

# ============ PAGE FUNCTIONS ============ #

def Cover():
    # 3 kolom: logo1 | logo2 | kosong (untuk dorong semua ke kiri)
    col1, col2, col3 = st.columns([1, 1, 6])  # total 8 bagian, kolom 3 untuk kosongkan kanan

    with col1:
        if os.path.exists("assets/logo_its.png"):
            st.image("assets/logo_its.png", width=80)
        else:
            st.write("")

    with col2:
        if os.path.exists("assets/logo_bme.png"):
            st.image("assets/logo_bme.png", width=80)
        else:
            st.write("")

    # Judul dan informasi utama
    st.title("TUGAS AKHIR")
    st.header("KLASIFIKASI TINGKAT KEPARAHAN GLAUKOMA BERDASARKAN FITUR MORFOLOGI PADA CITRA FUNDUS RETINA MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK (CNN)")
    st.subheader("Nadhifatul Fuadah - 5023211053")
    st.markdown("### Dosen Pembimbing 1: Dr. Tri Arief Sardjono, S.T., M.T")
    st.markdown("### Dosen Pembimbing 2: Nada Fitrieyatul Hikmah, S.T., M.T")

    # Sidebar
    st.sidebar.info(
        "Navigation Instructions:\n"
        "- Go to **Preprocessing** to enhance image quality\n"
        "- Go to **Segmentation** to choose between OD/OC or Vessel\n"
        "- Go to **Feature Extraction** to analyze CDR, vessel tortuosity, etc.\n"
        "- Use **Classification** to predict glaucoma severity\n"
        "- Visit **About Glaucoma** to learn more"
    )


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

def Preprocessing():
    st.title("Preprocessing")
    st.markdown("Upload a fundus image to apply preprocessing (CLAHE, normalization, etc.)")
    uploaded_file = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Fundus Image", use_column_width=True)
        st.success("Image uploaded. You can now apply preprocessing.")

def Segmentation():
    st.title("Segmentation")
    st.markdown("Select segmentation model:")
    seg_type = st.selectbox("Segmentation Type", ["Optic Disc & Cup", "Blood Vessel"])

    if seg_type == "Optic Disc & Cup":
        st.markdown("Running OD/OC segmentation model...")
        # You can place your code or button here
    elif seg_type == "Blood Vessel":
        st.markdown("Running vessel segmentation model...")
        # You can place your code or button here

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

# ============ SIDEBAR SELECTOR ============ #

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

# ============ PAGE ROUTER ============ #

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
