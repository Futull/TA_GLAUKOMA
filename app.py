import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="TUGAS AKHIR",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .sidebar .sidebar-content { padding: 2rem 1rem; }
    .block-container { padding: 2rem 2rem; }
    h1, h2, h3 { color: #1c1c1c; }
    .stButton>button {
        background-color: #004d99;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", [
    "Home", 
    "About Glaucoma", 
    "Preprocessing", 
    "Segmentation", 
    "Feature Extraction", 
    "Classification", 
    "Evaluation"
])

# Home Page
if page == "Home":
    st.title("Final Project")
    st.markdown("""
    ## KLASIFIKASI TINGKAT KEPARAHAN GLAUKOMA BERDASARKAN FITUR MORFOLOGI PADA CITRA FUNDUS RETINA MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK (CNN)
    **Name**: [NADHIFATUL FUADAH]  
    **NRP** : 5023211053  
    **Dosen Pembimbing 1**: Dr. Tri Arief Sardjono, S.T., M.T  
    **Dosen Pembimbing 2**: Nada Fitrieyatul Hikmah, S.T., M.T
    """)

# About Glaucoma Page
elif page == "About Glaucoma":
    st.title("About Glaucoma")
    st.markdown("""
    **What is Glaucoma?**  
    Glaucoma is a group of eye conditions that damage the optic nerve, often due to high intraocular pressure. It is one of the leading causes of irreversible blindness.

    **Severity Stages:**
    - **Normal**: No optic nerve damage.
    - **Mild**: Slight cupping and early signs.
    - **Moderate**: Noticeable damage and vision loss.
    - **Severe**: Advanced damage with high blindness risk.

    **Why Morphological Analysis?**  
    Morphological changes in the optic disc and blood vessels offer strong indicators for glaucoma severity, such as:
    - Cup-to-disc ratio (CDR)
    - Blood vessel tortuosity
    - Bifurcation patterns
    - Vessel skeletonization

    This system integrates these to build a deep learning-based classification tool.
    """)

# Preprocessing Page
elif page == "Preprocessing":
    st.header("Preprocessing")
    st.markdown("Upload a retinal fundus image to perform normalization, CLAHE, resizing, etc.")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success("Ready for preprocessing.")

# Segmentation Page
elif page == "Segmentation":
    st.header("Segmentation")
    seg_option = st.radio("Choose segmentation type:", ["Optic Disc & Cup", "Blood Vessel"])
    
    if seg_option == "Optic Disc & Cup":
        st.markdown("Segmentation of OD/OC using model 1...")
        # Add upload & predict logic for OD/OC
    elif seg_option == "Blood Vessel":
        st.markdown("Segmentation of vessels using model 2...")
        # Add upload & predict logic for vessels

# Feature Extraction Page
elif page == "Feature Extraction":
    st.header("Feature Extraction")
    feat_option = st.radio("Choose feature type:", ["OD/OC Features", "Vessel Features"])

    if feat_option == "OD/OC Features":
        st.markdown("Extracting features such as CDR, area, eccentricity, solidity, etc.")
        # Add logic or visual here
    elif feat_option == "Vessel Features":
        st.markdown("Extracting vessel-based features like tortuosity, skeleton length, bifurcation points, etc.")
        # Add logic or visual here

# Classification Page
elif page == "Classification":
    st.header("Classification with Support Vector Machine")
    # Add model call and result display

# Evaluation Page
elif page == "Evaluation":
    st.header("Evaluation and Results")
    st.markdown("Show model metrics such as confusion matrix, accuracy, F1-score, sensitivity, etc.")

