import streamlit as st
from PIL import Image      
import os    
from skimage.measure import label, regionprops   
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes, skeletonize
from skimage.filters import gaussian  
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
import cv2  
import numpy as np   
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import euclidean
import pandas as pd
from skimage.morphology import skeletonize
import networkx as nx
import math
from model_architecture import Build_UNet  # OD/OC
from vessel_architecture import Build_UNet_Vessel  # VESSEL
import pickle
import gdown

# ===================== MODEL DOWNLOADER (GDRIVE) ===================== #

def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def ensure_models_exist():
    model_files = [
        {"file_id": "1AaNBCHRaLJ5mtIWG9bc0dtvDjD9Z1pj7", "dest_path": "models/fix_model_odoc.pt"},
        {"file_id": "1vDhVgOmJrZslRQ_j_qtE-UuBNNn5aksD", "dest_path": "models/fix_model_vessel.pt"},
    ]
    for m in model_files:
        if not os.path.exists(m["dest_path"]) or os.stat(m["dest_path"]).st_size == 0:
            # Download without any Streamlit output
            os.makedirs(os.path.dirname(m["dest_path"]), exist_ok=True)
            download_from_gdrive(m["file_id"], m["dest_path"])
            size = os.path.getsize(m["dest_path"])
            # Optional: bisa tambahkan print untuk debugging lokal (hapus jika tidak mau output apapun)
            # print(f"Model {os.path.basename(m['dest_path'])} size: {size/1024/1024:.2f} MB")
            if size < 10000:
                # print(f"File {m['dest_path']} terlalu kecil, kemungkinan gagal download (HTML/Quota).")
                with open(m["dest_path"], 'rb') as f:
                    head = f.read(200)
                # print(head)
            # else:
                # print(f"Download {os.path.basename(m['dest_path'])} selesai!")

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
        "Application Guide:\n"
        "- Go to *DETECTION* page to start the analysis\n"
        "- Follow the sequential steps: Preprocessing → Segmentation → Feature Extraction → Classification\n"
        "- Upload your fundus images to begin the detection process\n"
    )

# ===================== RESET FUNCTIONS ===================== #
def reset_od_oc_pipeline():
    od_oc_keys = [
        'preprocessing_results_od', 'preprocessed_image_od', 'original_image_od',
        'segmentation_map_od', 'colored_segmentation_od', 'segmentation_completed_od',
        'extracted_features_od', 'current_od_image_hash'
    ]
    for key in od_oc_keys:
        if key in st.session_state:
            del st.session_state[key]

def reset_vessel_pipeline():
    vessel_keys = [
        'preprocessing_results_vessel', 'preprocessed_image_vessel', 'original_image_vessel',
        'vessel_mask', 'segmentation_completed_vessel', 
        'extracted_features_vessel', 'current_vessel_image_hash'
    ]
    for key in vessel_keys:
        if key in st.session_state:
            del st.session_state[key]

def reset_classification():
    classification_keys = [
        'extracted_features', 'classification_result'
    ]
    for key in classification_keys:
        if key in st.session_state:
            del st.session_state[key]

def reset_extraction_and_classification():
    extraction_classification_keys = [
        'extracted_features', 'classification_result', 
        'extracted_features_od', 'extracted_features_vessel'
    ]
    for key in extraction_classification_keys:
        if key in st.session_state:
            del st.session_state[key]
    if 'step_completed' in st.session_state:
        st.session_state.step_completed['extraction'] = False

def reset_all_steps():
    st.session_state.step_completed = {'preprocessing': False, 'segmentation': False, 'extraction': False}
    reset_od_oc_pipeline()
    reset_vessel_pipeline()
    reset_classification()

def get_image_hash(image_array):
    return hash(image_array.tobytes())

# ===================== PREPROCESSING FUNCTIONS ===================== #

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def unsharp_mask(image, blur_ksize=5, strength=1.0):
    blur = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    mask = cv2.addWeighted(image, 1 + strength, blur, -strength, 0)
    return np.clip(mask, 0, 255).astype(np.uint8)

def high_pass_filter(image):
    low_pass = cv2.GaussianBlur(image, (9, 9), 0)
    high_pass = cv2.subtract(image, low_pass)
    sharpened = cv2.add(image, high_pass)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def combined_sharpening(image):
    unsharp = unsharp_mask(image, blur_ksize=5, strength=1.5)
    highpass = high_pass_filter(unsharp)
    return highpass

def color_normalization_fixed(image, avg_r=0.9601, avg_g=0.6374, avg_b=0.3408):
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
    normalized = image / 255.0
    corrected = np.power(normalized, 1.0 / gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(12, 12)):
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
    return cv2.medianBlur(image, ksize)
    
def preprocess_od_oc_stepwise(image):
    results = {}
    resized_image = resize_image(image, target_size=(256, 256))
    results['step1_resized'] = resized_image
    sharpened_image = combined_sharpening(resized_image)
    results['step2_sharpened'] = sharpened_image
    color_normalized_image = color_normalization_fixed(sharpened_image)
    results['step3_color_norm'] = color_normalized_image
    gamma_corrected_image = apply_gamma_correction(color_normalized_image, gamma=1.1)
    results['step4_gamma'] = gamma_corrected_image
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=2.0, tile_grid_size=(12, 12))
    results['step5_clahe'] = clahe_image
    final_image = apply_median_filter(clahe_image, ksize=3)
    results['step6_final'] = final_image
    return results

def preprocess_vessel_stepwise(image):
    results = {}
    resized_image = resize_image(image, target_size=(256, 256))
    results['step1_resized'] = resized_image
    if len(resized_image.shape) == 3:
        green_channel = resized_image[:, :, 1]
    else:
        green_channel = resized_image
    green_3ch = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2RGB)
    results['step2_green'] = green_3ch
    gamma_corrected_image = apply_gamma_correction(green_3ch, gamma=1.1)
    results['step3_gamma'] = gamma_corrected_image
    clahe_image = apply_clahe(gamma_corrected_image, clip_limit=2.0, tile_grid_size=(8, 8))
    results['step4_clahe'] = clahe_image
    final_image = apply_median_filter(clahe_image, ksize=3)
    results['step5_final'] = final_image
    return results

# ===================== SEGMENTATION FUNCTIONS ===================== #

def load_od_oc_model():
    try:
        model = Build_UNet()
        model.load_state_dict(torch.load('models/fix_model_odoc.pt', map_location='cpu', weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading OD/OC model: {e}")
        return None

def load_vessel_model():
    try:
        model = Build_UNet_Vessel()
        model.load_state_dict(torch.load('models/fix_model_vessel.pt', map_location='cpu', weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading vessel model: {e}")
        return None

def predict_od_oc(model, image):
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
    if len(image.shape) == 3:
        green_channel = image[:, :, 1]
    else:
        green_channel = image
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

# ===================== FEATURE EXTRACTION FUNCTIONS ===================== #

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

def extract_od_oc_features(mask):
    final_mask, disc_mask, cup_mask = postprocess_mask(mask)
    disc_props = regionprops(label(disc_mask))
    cup_props = regionprops(label(cup_mask))
    if not disc_props or not cup_props:
        return None
    disc = disc_props[0]
    cup = cup_props[0]
    h_cup = cup.bbox[2] - cup.bbox[0]
    h_disc = disc.bbox[2] - disc.bbox[0]
    cdr_vertical = h_cup / (h_disc + 1e-8)
    cdr_area = cup.area / (disc.area + 1e-8)
    cdr_diameter = cup.major_axis_length / (disc.major_axis_length + 1e-8)
    return {
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
    }

# ===================== OD/OC GLCM =====================
def extract_glcm_features_odoc(image, levels=32):
    green = image[:, :, 1]
    image_ubyte = img_as_ubyte(green)
    image_quantized = np.clip((image_ubyte / (256 // levels)).astype(np.uint8), 0, levels - 1)

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_quantized, distances=[1], angles=angles,
                        levels=levels, symmetric=True, normed=True)

    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    features = {}
    for prop in props:
        values = graycoprops(glcm, prop)[0]
        features[f"mean_{prop}_cdr"] = np.mean(values)  # OD/OC suffix
    return features

# ===================== VESSEL GLCM =====================
def extract_glcm_features_vessel(image, levels=32):
    # image: 2D (grayscale)
    image_ubyte = img_as_ubyte(image)
    image_quantized = np.clip((image_ubyte / (256 // levels)).astype(np.uint8), 0, levels - 1)

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_quantized, distances=[1], angles=angles,
                        levels=levels, symmetric=True, normed=True)
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    features = {}
    for prop in props:
        values = graycoprops(glcm, prop)[0]
        features[f"mean_{prop}_vessel"] = np.mean(values)
    return features

def prepare_vessel_mask(mask):
        # Jika mask 3D, ambil channel pertama (umumnya [H, W, 3])
        if mask.ndim == 3:
            if mask.shape[0] == 3 and mask.shape[2] != 3:
                # (3, H, W) -> ambil [0]
                mask = mask[0]
            else:
                # (H, W, 3) -> ambil [...,0]
                mask = mask[..., 0]
        bin_mask = (mask > 0).astype(np.uint8)
        return bin_mask

def generate_vessel_skeleton(bin_mask):
    # Skeletonisasi dari mask biner
    skeleton = skeletonize(bin_mask).astype(np.uint8)
    return skeleton

def build_graph(skel):
    G = nx.Graph()
    h, w = skel.shape
    for y in range(h):
        for x in range(w):
            if skel[y, x]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx_ = y + dy, x + dx
                        if (dy != 0 or dx != 0) and 0 <= ny < h and 0 <= nx_ < w:
                            if skel[ny, nx_]:
                                G.add_edge((y, x), (ny, nx_))
    return G

# =========== TORTUOSITY =============#
def extract_tortuosity_features_from_skeleton(skeleton):
    G = build_graph(skeleton)
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    branches = [n for n in G.nodes if G.degree[n] >= 3]
    important_points = set(endpoints + branches)

    visited = set()
    segments = []
    for node in important_points:
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            if (node, neighbor) in visited or (neighbor, node) in visited:
                continue
            path = [node, neighbor]
            current = neighbor
            prev = node
            while current not in important_points:
                next_nodes = list(G.neighbors(current))
                if prev in next_nodes:
                    next_nodes.remove(prev)
                if not next_nodes:
                    break
                prev, current = current, next_nodes[0]
                path.append(current)
            if len(path) > 2:
                segments.append(path)
                for i in range(len(path) - 1):
                    visited.add((path[i], path[i+1]))

    tortuosity_list = []
    for seg in segments:
        s_length = sum(euclidean(seg[i], seg[i+1]) for i in range(len(seg) - 1))
        s_straight = euclidean(seg[0], seg[-1])
        if s_straight > 0:
            TC = s_length / s_straight
            if TC < 10:
                tortuosity_list.append(TC)

    return {
        "Mean Tortuosity": round(np.mean(tortuosity_list), 4) if tortuosity_list else 0,
        "Median Tortuosity": round(np.median(tortuosity_list), 4) if tortuosity_list else 0,
        "Std Dev TC": round(np.std(tortuosity_list), 4) if tortuosity_list else 0,
        "Number of segments": len(tortuosity_list)
    }

# ============ BIFURCATION POINT =========== #
def get_direction_vectors(y, x, img):
    directions = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                if img[ny, nx] == 1:
                    directions.append((dy, dx))
    return directions

def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    norm1 = math.hypot(*v1)
    norm2 = math.hypot(*v2)
    cos_theta = dot / (norm1 * norm2 + 1e-6)
    return math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))

def extract_bifurcation_features_from_skeleton(skeleton):
    bif_points = []
    for y in range(1, skeleton.shape[0]-1):
        for x in range(1, skeleton.shape[1]-1):
            if skeleton[y, x] == 1:
                neighbors = get_direction_vectors(y, x, skeleton)
                if len(neighbors) == 3:
                    angles = [
                        angle_between(neighbors[i], neighbors[j])
                        for i in range(3) for j in range(i+1, 3)
                    ]
                    if max(angles) - min(angles) > 40:
                        bif_points.append((x, y))
    return {"Bifurcation Point": len(bif_points)}

# ============ VESSEL LENGTH ===========#
def compute_vessel_length(skeleton):
    coords = np.column_stack(np.where(skeleton > 0))
    visited = set()
    length = 0.0
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0), (1, 1)]
    for y, x in coords:
        for dy, dx in neighbor_offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                if skeleton[ny, nx] == 1:
                    edge = tuple(sorted([(y, x), (ny, nx)]))
                    if edge not in visited:
                        dist = np.sqrt((ny - y)**2 + (nx - x)**2)
                        length += dist
                        visited.add(edge)
    return length

def extract_vessel_length_features_from_skeleton(skeleton):
    return {
        "Vessel_Length": round(compute_vessel_length(skeleton), 2)
    }

# ===================== VESSEL AREA & DENSITY =============#
def extract_vessel_area_density_features(bin_mask):
    Vessel_Area = np.sum(bin_mask)
    total_area = bin_mask.shape[0] * bin_mask.shape[1]
    Vessel_Density = Vessel_Area / total_area
    return {
        "Vessel_Area": int(Vessel_Area),
        "Vessel_Density": round(Vessel_Density, 6)
    }

@st.cache_resource
def load_svm_classifier():
    with open("models/finall_svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/finall_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/finall_top10_features.pkl", "rb") as f:
        selected_features = pickle.load(f)
    return model, scaler, selected_features

# ===================== DETECTION PAGE ===================== #

def Detection():
    st.title("👁️‍🗨️ Glaucoma Detection System")
    st.markdown("---")
    if 'step_completed' not in st.session_state:
        st.session_state.step_completed = {'preprocessing': False, 'segmentation': False, 'extraction': False}
    state_keys = [
        'preprocessing_results_od', 'preprocessed_image_od', 'original_image_od',
        'preprocessing_results_vessel', 'preprocessed_image_vessel', 'original_image_vessel',
        'segmentation_map_od', 'colored_segmentation_od', 'segmentation_completed_od',
        'vessel_mask', 'segmentation_completed_vessel', 'extracted_features', 'classification_result',
        'extracted_features_od', 'extracted_features_vessel', 'current_od_image_hash', 'current_vessel_image_hash'
    ]
    for key in state_keys:
        if key not in st.session_state:
            st.session_state[key] = None
    col_reset1, col_reset2, col_reset3 = st.columns([2, 1, 2])
    with col_reset2:
        if st.button("🔄 RESET ALL", key="reset_all", type="secondary", use_container_width=True):
            reset_all_steps()
            st.rerun()
            
    # STEP 1: PREPROCESSING
    st.header("⚙️ Step 1: PREPROCESSING ")
    st.markdown("Upload your fundus images and select the preprocessing pipeline based on your analysis needs.")
    
    # Image upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔘 Optic Disc & Cup (OD/OC)")
        st.markdown("*For Cup-to-Disc Ratio analysis*")
        uploaded_file_od = st.file_uploader("Upload Cropped ONH Image", type=["png", "jpg", "jpeg"], key="od_oc_upload")
        
        if uploaded_file_od:
            image_od = Image.open(uploaded_file_od).convert('RGB')
            img_np_od = np.array(image_od)
            current_hash = get_image_hash(img_np_od)
            
            # Check if image has changed
            if st.session_state.get('current_od_image_hash') != current_hash:
                reset_od_oc_pipeline()
                reset_extraction_and_classification()  # PERBAIKAN: Reset extraction dan classification
                st.session_state['current_od_image_hash'] = current_hash
                st.rerun()
            
            st.image(img_np_od, caption="Original Cropped ONH Image", use_container_width=True)
            
            if st.button("🟢 Apply OD/OC Preprocessing", key="preprocess_od_oc"):
                with st.spinner("Processing OD/OC preprocessing..."):
                    results = preprocess_od_oc_stepwise(img_np_od)
                    st.session_state['preprocessing_results_od'] = results
                    st.session_state['preprocessed_image_od'] = results['step6_final']
                    st.session_state['original_image_od'] = img_np_od
                    st.session_state.step_completed['preprocessing'] = True
                    # PERBAIKAN: Reset extraction dan classification ketika preprocessing ulang
                    reset_extraction_and_classification()
                    st.success("✅ OD/OC preprocessing completed!")
        
        # Always show results if they exist in session state
        if st.session_state.get('preprocessing_results_od') is not None:
            results = st.session_state['preprocessing_results_od']
            original_img = st.session_state['original_image_od']
            
            st.subheader("🔄 OD/OC Preprocessing Pipeline Results")
            cols = st.columns(4)
            
            with cols[0]:
                st.image(original_img, caption="Original")
            with cols[1]:
                st.image(results['step1_resized'], caption="1. Resized")
            with cols[2]:
                st.image(results['step2_sharpened'], caption="2. Sharpened")
            with cols[3]:
                st.image(results['step3_color_norm'], caption="3. Color Norm")
            
            cols2 = st.columns(3)
            with cols2[0]:
                st.image(results['step4_gamma'], caption="4. Gamma Correct")
            with cols2[1]:
                st.image(results['step5_clahe'], caption="5. CLAHE")
            with cols2[2]:
                st.image(results['step6_final'], caption="6. Final Result")
    
    with col2:
        st.subheader("🩸 Retina Vessel")
        st.markdown("*For Retina Blood Vessel Morphology Analysis*")
        uploaded_file_vessel = st.file_uploader("Upload Full Fundus Image", type=["png", "jpg", "jpeg"], key="vessel_upload")
        
        if uploaded_file_vessel:
            image_vessel = Image.open(uploaded_file_vessel).convert('RGB')
            img_np_vessel = np.array(image_vessel)
            current_hash = get_image_hash(img_np_vessel)
            
            # Check if image has changed
            if st.session_state.get('current_vessel_image_hash') != current_hash:
                reset_vessel_pipeline()
                reset_extraction_and_classification()  # PERBAIKAN: Reset extraction dan classification
                st.session_state['current_vessel_image_hash'] = current_hash
                st.rerun()
            
            st.image(img_np_vessel, caption="Original Full Fundus Image", use_container_width=True)
            
            if st.button("🟢 Apply Vessel Preprocessing", key="preprocess_vessel"):
                with st.spinner("Processing vessel preprocessing..."):
                    results = preprocess_vessel_stepwise(img_np_vessel)
                    st.session_state['preprocessing_results_vessel'] = results
                    st.session_state['preprocessed_image_vessel'] = results['step5_final']
                    st.session_state['original_image_vessel'] = img_np_vessel
                    st.session_state.step_completed['preprocessing'] = True
                    # PERBAIKAN: Reset extraction dan classification ketika preprocessing ulang
                    reset_extraction_and_classification()
                    st.success("✅ Vessel preprocessing completed!")
        
        # Always show results if they exist in session state
        if st.session_state.get('preprocessing_results_vessel') is not None:
            results = st.session_state['preprocessing_results_vessel']
            original_img = st.session_state['original_image_vessel']
            
            st.subheader("🔄 Vessel Preprocessing Pipeline Results")
            cols = st.columns(3)
            
            with cols[0]:
                st.image(original_img, caption="Original")
            with cols[1]:
                st.image(results['step1_resized'], caption="1. Resized")
            with cols[2]:
                st.image(results['step2_green'], caption="2. Green Channel")
            
            cols2 = st.columns(3)
            with cols2[0]:
                st.image(results['step3_gamma'], caption="3. Gamma Correct")
            with cols2[1]:
                st.image(results['step4_clahe'], caption="4. CLAHE")
            with cols2[2]:
                st.image(results['step5_final'], caption="5. Final Result")
    
    st.markdown("---")
    
    # STEP 2: SEGMENTATION
    # Check if any preprocessing is completed
    has_od_preprocessing = st.session_state.get('preprocessed_image_od') is not None
    has_vessel_preprocessing = st.session_state.get('preprocessed_image_vessel') is not None
    
    if has_od_preprocessing or has_vessel_preprocessing:
        st.header("⚙️ Step 2: SEGMENTATION")
        st.markdown("Perform automatic segmentation using trained deep learning models.")
        
        segmentation_cols = st.columns(2)
        
        # OD/OC Segmentation
        if has_od_preprocessing:
            with segmentation_cols[0]:
                st.subheader("🔘 OD/OC Segmentation")
                preprocessed_img_od = st.session_state['preprocessed_image_od']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(preprocessed_img_od, caption="Preprocessed Image")
                
                # Show existing segmentation result if available
                if st.session_state.get('segmentation_completed_od'):
                    with col2:
                        st.image(st.session_state['colored_segmentation_od'], caption="OD/OC Segmentation")
                        st.success("✅ OD/OC segmentation completed!")
                
                if st.button("🟢 Run OD/OC Segmentation", key="segment_od_oc"):
                    with st.spinner("Running OD/OC segmentation..."):
                        model = load_od_oc_model()
                        
                        if model is not None:
                            prediction = predict_od_oc(model, preprocessed_img_od)
                            
                            if len(prediction.shape) == 3 and prediction.shape[0] > 1:
                                seg_map = np.argmax(prediction, axis=0)
                            else:
                                seg_map = prediction.squeeze()
                            
                            # Create colored visualization
                            colored_result = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
                            colored_result[seg_map == 1] = [169, 169, 169]  # OD = Grey
                            colored_result[seg_map == 2] = [255, 255, 255]  # OC = White
                            
                            # Save results to session state
                            st.session_state['segmentation_map_od'] = seg_map
                            st.session_state['colored_segmentation_od'] = colored_result
                            st.session_state['segmentation_completed_od'] = True
                            st.session_state.step_completed['segmentation'] = True
                            
                            # PERBAIKAN: Reset extraction dan classification ketika segmentasi baru
                            reset_extraction_and_classification()
                            
                            with col2:
                                st.image(colored_result, caption="OD/OC Segmentation")
                            
                            st.success("✅ OD/OC segmentation completed!")
        
        # Vessel Segmentation
        if has_vessel_preprocessing:
            with segmentation_cols[1]:
                st.subheader("🩸 Retina Vessel Segmentation")
                preprocessed_img_vessel = st.session_state['preprocessed_image_vessel']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(preprocessed_img_vessel, caption="Preprocessed Image")
                
                # Show existing segmentation result if available
                if st.session_state.get('segmentation_completed_vessel'):
                    with col2:
                        st.image(st.session_state['vessel_mask'], caption="Retina Vessel Segmentation")
                        st.success("✅ Vessel Segmentation Completed!")
                
                if st.button("🟢 Run Retina Vessel Segmentation", key="segment_vessel"):
                    with st.spinner("Running Retina Vessel Segmentation..."):
                        model = load_vessel_model()
                        
                        if model is not None:
                            prediction = predict_vessel(model, preprocessed_img_vessel)
                            
                            # Convert prediction to displayable format
                            vessel_mask = (prediction > 0.5).astype(np.uint8) * 255
                            if len(vessel_mask.shape) == 3:
                                vessel_mask = vessel_mask[0]
                            
                            # Save results to session state
                            st.session_state['vessel_mask'] = vessel_mask
                            st.session_state['segmentation_completed_vessel'] = True
                            st.session_state.step_completed['segmentation'] = True
                            
                            # PERBAIKAN: Reset extraction dan classification ketika segmentasi baru
                            reset_extraction_and_classification()
                            
                            with col2:
                                st.image(vessel_mask, caption="Vessel Segmentation")
                            
                            st.success("✅ Vessel segmentation completed!")
        
        st.markdown("---")
    
    # STEP 3: FEATURE EXTRACTION
    # Check if any segmentation is completed
    has_od_segmentation = st.session_state.get('segmentation_completed_od')
    has_vessel_segmentation = st.session_state.get('segmentation_completed_vessel')
    
    if has_od_segmentation or has_vessel_segmentation:
        st.header("⚙️ Step 3: FEATURE EXTRACTION")
        st.markdown("Extract morphological features from segmentation results for classification.")
        
        extraction_cols = st.columns(2)
        
        # Feature extraction info
        with extraction_cols[0]:
            st.subheader("🔍 Available Features")
            if has_od_segmentation:
                st.markdown("""
                **OD/OC Features:**
                - Cup-to-Disc Ratio (CDR)
                - Disc & Cup morphological properties
                - GLCM texture features
                """)
            
            if has_vessel_segmentation:
                st.markdown("""
                **Vessel Features:**
                - Vessel tortuosity analysis
                - Bifurcation points detection
                - Vessel length, area, & density
                - GLCM texture features
                """)
        
        with extraction_cols[1]:
            st.subheader("🟢 Start Extraction")
            
            # Show existing features if available
            if st.session_state.get('extracted_features') is not None:
                st.success("✅ Features already extracted!")
                st.info("Features are ready for classification.")
            
            if st.button("START FEATURE EXTRACTION", key="extract_features", type="primary"):
                with st.spinner("Extracting features from segmentation results..."):
                    extracted_features = {}
                    
                    # Extract OD/OC features
                    if has_od_segmentation:
                        st.write("🔘 Extracting OD/OC Features...")
                        
                        # Regionprops features
                        seg_map = st.session_state['segmentation_map_od']
                        od_oc_features = extract_od_oc_features(seg_map)
                        
                        if od_oc_features:
                            extracted_features.update(od_oc_features)
                            st.session_state['extracted_features_od'] = od_oc_features
                        
                        # GLCM features from preprocessed image
                        preprocessed_img = st.session_state['preprocessed_image_od']
                        glcm_features_od = extract_glcm_features_odoc(preprocessed_img)
                        
                        glcm_features_od = extract_glcm_features_odoc(preprocessed_img)
                        extracted_features.update(glcm_features_od)

                    
                    # ================Extract Vessel features=============#
                    # Vessel Features
                if has_vessel_segmentation:
                    st.write("🩸 Extracting Retina Vessel Features...")
                    vessel_mask = st.session_state['vessel_mask']
                    preprocessed_vessel_img = st.session_state['preprocessed_image_vessel']

                    vessel_features = {}

                    bin_mask = prepare_vessel_mask(vessel_mask)
                    skeleton = skeletonize(bin_mask).astype(np.uint8)

                    # --- Ekstraksi fitur GLCM hanya dari satu channel green ---
                    if preprocessed_vessel_img.ndim == 3:
                        green_img = preprocessed_vessel_img[:, :, 1]
                    else:
                        green_img = preprocessed_vessel_img
                    glcm_features_vessel = extract_glcm_features_vessel(green_img, levels=32)
   
                    vessel_features.update(glcm_features_vessel)

                    # Tortuosity, bifurcation, vessel length, area/density dari mask/skeleton biner
                    vessel_features.update(extract_tortuosity_features_from_skeleton(skeleton))
                    vessel_features.update(extract_bifurcation_features_from_skeleton(skeleton))
                    vessel_features.update(extract_vessel_length_features_from_skeleton(skeleton))
                    vessel_features.update(extract_vessel_area_density_features(bin_mask))

                    # Update ke state Streamlit
                    extracted_features.update(vessel_features)
                    st.session_state['extracted_features_vessel'] = vessel_features

                st.session_state['extracted_features'] = extracted_features
                st.session_state.step_completed['extraction'] = True
                st.success("✅ Feature extraction completed!")
        
        # Display extracted features table if available
        if st.session_state.get('extracted_features') is not None:
            st.subheader("📋 Extracted Features Summary")
            
            features_df = pd.DataFrame([st.session_state['extracted_features']])
            
            # Display features in a more organized way
            if has_od_segmentation:
                st.markdown("**🔘 OD/OC Features:**")
                od_oc_cols = [col for col in features_df.columns if any(x in col.lower() for x in ['cdr', 'cup', 'disc', '_cdr'])]
                if od_oc_cols:
                    st.dataframe(features_df[od_oc_cols], use_container_width=True)
            
            if has_vessel_segmentation:
                st.markdown("**🩸 Retina Vessel Features:**")
                vessel_cols = [
                    col for col in features_df.columns
                    if (
                        col.endswith('_vessel') or
                        col.startswith('Vessel_') or
                        col.startswith('Mean Tortuosity') or
                        col.startswith('Median Tortuosity') or
                        col.startswith('StdDev Tortuosity') or
                        col.startswith('Num Tortuosity') or
                        col == 'Bifurcation Point'
                    )
                ]

                if vessel_cols:
                    st.dataframe(features_df[vessel_cols], use_container_width=True)
            
            st.markdown("**📊 Complete All Features :**")
            st.dataframe(features_df, use_container_width=True)
        
        st.markdown("---")

    # ===================== STEP 4: CLASSIFICATION ===================== #
    if st.session_state.get('extracted_features') is not None:
        st.header("⚙️ Step 4: CLASSIFICATION")
        st.markdown("Classify glaucoma severity based on extracted features using trained SVM model.")

        model, scaler, selected_features = load_svm_classifier()
        extracted = st.session_state['extracted_features']

        missing = set(selected_features) - set(extracted.keys())
        extra = set(extracted.keys()) - set(selected_features)

        if st.button("🧠 Run Classification", key="run_classification", type="primary"):
            with st.spinner("Running SVM classification..."):
                # Buat dict lokal untuk mapping, supaya session_state tidak berubah
                features_to_classify = extracted.copy()  # atau extracted_features.copy()
                
                # Cek dan isi fitur yang missing (jika ada)
                for f in selected_features:
                    if f not in features_to_classify:
                        features_to_classify[f] = 0  # atau np.nan
                
                # Susun input_vector sesuai urutan selected_features
                input_vector = [features_to_classify[f] for f in selected_features]
                
                # Lanjutkan scaling & prediksi
                input_array = np.array(input_vector).reshape(1, -1)
                scaled_input = scaler.transform(input_array)
                prediction = model.predict(scaled_input)[0]
                probabilities = model.predict_proba(scaled_input)[0]
        
                # ======== END DEBUG ========
        
                # (3) Prediksi
                prediction = model.predict(scaled_input)[0]
                probabilities = model.predict_proba(scaled_input)[0]

                label_map = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
                result_label = label_map.get(prediction, "Unknown")

                st.session_state['classification_result'] = result_label

                st.success(f"🧠 Predicted Glaucoma Severity: **{result_label}**")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Progress Tracker")
        progress_items = [
            ("Preprocessing", st.session_state.step_completed['preprocessing']),
            ("Segmentation", st.session_state.step_completed['segmentation']),
            ("Feature Extraction", st.session_state.step_completed['extraction']),
            ("Classification", st.session_state.get('classification_result') is not None)
        ]
        for step, completed in progress_items:
            if completed:
                st.sidebar.markdown(f"✅ {step}")
            else:
                st.sidebar.markdown(f"⏳ {step}")
        if st.sidebar.checkbox("🔧 Debug Info"):
            st.sidebar.write("**Current States:**")
            st.sidebar.write(f"OD Hash: {st.session_state.get('current_od_image_hash', 'None')}")
            st.sidebar.write(f"Vessel Hash: {st.session_state.get('current_vessel_image_hash', 'None')}")
            st.sidebar.write(f"Features Count: {len(st.session_state.get('extracted_features', {}))}")
            if st.session_state.get('classification_result') and 'image_hashes' in st.session_state['classification_result']:
                class_hashes = st.session_state['classification_result']['image_hashes']
                st.sidebar.write("**Classification Image Hashes:**")
                st.sidebar.write(f"OD: {class_hashes.get('od_hash', 'None')}")
                st.sidebar.write(f"Vessel: {class_hashes.get('vessel_hash', 'None')}")

# ===================== PAGE ROUTING ===================== #

def main():
    st.set_page_config(
        page_title="Glaucoma Severity Detection System",
        page_icon="👁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    ensure_models_exist()
    st.sidebar.title("👁 Glaucoma Severity Detection System")
    page = st.sidebar.selectbox("NAVIGATION", [
        "COVER", 
        "DETECTION"
    ])
    if page == "COVER":
        Cover()
    elif page == "DETECTION":
        Detection()

if __name__ == "__main__":
    main()
