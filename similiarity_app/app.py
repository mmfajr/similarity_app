import streamlit as st
import cv2
import numpy as np
import os
import plotly.graph_objects as go
import time

# --- KODE UNTUK MENGUBAH TAMPILAN (BACKGROUND & FONT) ---
# Versi ini menggunakan selector yang lebih kuat untuk memastikan gaya diterapkan.
st.markdown("""
<style>
/* Target elemen utama tempat aplikasi Streamlit dirender */
.stApp {
    background-image: linear-gradient(to bottom right, #0b192f, #1e3a5f);
    color: white;
}

/* Mengubah warna teks secara umum */
h1, h2, h3, h4, h5, h6, p, li, label {
    color: #FFFFFF !important;
}

/* Styling untuk st.metric */
div[data-testid="stMetricLabel"] { color: #a0a8b3 !important; }
div[data-testid="stMetricValue"] { color: #FFFFFF !important; }

/* Memberi background pada kolom agar terlihat terpisah */
[data-testid="stVerticalBlock"] {
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    padding: 1.5rem !important;
    margin-bottom: 1rem;
}

/* Menghilangkan background putih pada beberapa widget */
[data-testid="stFileUploader"] {
    background-color: transparent;
}

/* --- Styling untuk File Uploader --- */
[data-testid="stFileUploader"] > label { color: #FFFFFF !important; }
div[data-testid="stFileUploaderFileName"] { color: #FFFFFF !important; }
[data-testid="stFileUploader"] section button {
    background-color: #007bff;
    color: white;
    border: none;
    transition: background-color 0.2s;
}
[data-testid="stFileUploader"] section button:hover {
    background-color: #0056b3;
}

/* 🎨 [BARU] Styling untuk Expander/Dropdown yang lebih baik */
[data-testid="stExpander"] {
    background-color: rgba(0, 0, 0, 0.2); /* Menyamakan bg dengan kolom */
    border: none !important; /* Menghilangkan border default */
    box-shadow: none !important;
    border-radius: 10px; /* Membuat sudut lebih tumpul */
}

[data-testid="stExpander"] summary {
    padding: 0.5rem 1rem;
    border-radius: 10px;
    /* Aturan hover dihilangkan agar warna tidak berubah */
}

[data-testid="stExpander"] summary p {
    color: #FFFFFF !important; /* Memastikan teks header selalu putih */
    font-weight: 600;
}

[data-testid="stExpander"] svg {
    color: #FFFFFF !important; /* Membuat ikon panah menjadi putih */
}
/* --- Akhir Styling Expander --- */

/* Menghilangkan background header utama */
[data-testid="stHeader"] {
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Similaritas Citra",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Fungsi Inti Pengolahan Citra (Tidak ada perubahan) ---

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing

def calculate_signature(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    main_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(main_contour)
    if M["m00"] == 0: return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = (cx, cy)
    distances = [np.sqrt((p[0][0] - cx)**2 + (p[0][1] - cy)**2) for p in main_contour]
    angles = [np.arctan2(p[0][1] - cy, p[0][0] - cx) for p in main_contour]
    sorted_indices = np.argsort(angles)
    sorted_distances = np.array(distances)[sorted_indices]
    signature = np.interp(np.linspace(0, len(sorted_distances) - 1, 360), np.arange(len(sorted_distances)), sorted_distances)
    return signature, centroid

def normalize_signature(signature):
    min_val, max_val = np.min(signature), np.max(signature)
    if max_val - min_val == 0: return np.zeros_like(signature)
    r_norm = (signature - min_val) / (max_val - min_val)
    start_index = np.argmin(r_norm)
    return np.roll(r_norm, -start_index)

def process_image_fully(image):
    binary = segment_image(image)
    signature, centroid = calculate_signature(binary)
    if signature is None: return None, None, None
    normalized_sig = normalize_signature(signature)
    return binary, normalized_sig, centroid

@st.cache_data
def load_reference_signatures(folder_path):
    ref_signatures = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            class_name = os.path.splitext(filename)[0].replace("_", " ").title()
            image_path = os.path.join(folder_path, filename)
            ref_image = cv2.imread(image_path)
            if ref_image is not None:
                _, normalized_sig, _ = process_image_fully(ref_image)
                if normalized_sig is not None:
                    ref_signatures[class_name] = normalized_sig
    return ref_signatures

def classify_image(test_signature, ref_signatures):
    distances = {name: np.linalg.norm(test_signature - ref_sig) for name, ref_sig in ref_signatures.items()}
    best_match = min(distances, key=distances.get)
    min_distance = distances[best_match]
    return best_match, min_distance, distances

# --- Tampilan Utama Aplikasi (UI) dengan Layout 3 Seksi ---

# Mendefinisikan 3 kolom utama untuk layout
col_left, col_center, col_right = st.columns([1.2, 2, 1.2])

# --- Seksi Kanan (Kontrol) ---
with col_left:
    st.header("Panel Kontrol")
    st.markdown("---")
    
    ref_signatures = load_reference_signatures("reference_images")
    
    if not ref_signatures:
        st.error("Folder `reference_images` tidak ditemukan atau kosong.")
    else:
        st.success(f"{len(ref_signatures)} kelas referensi berhasil dimuat.")
        with st.expander("Lihat Kelas Referensi"):
            # Menampilkan daftar kelas referensi dalam sebuah tabel sederhana
            st.table(ref_signatures.keys())

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Unggah Gambar", 
        type=["jpg", "jpeg", "png"],
        help="Pilih gambar objek untuk dianalisis."
    )

# --- Seksi Tengah (Perbandingan Citra) ---
with col_center:
    st.title("Analisis Similaritas Citra Digital")
    
    # Inisialisasi variabel di awal untuk menghindari NameError
    test_signature = None
    binary_image = None

    if uploaded_file is None:
        st.info("Selamat Datang! Silakan unggah sebuah gambar pada Panel Kontrol di sebelah kanan untuk memulai analisis.")
    else:
        st.header("Input & Hasil Segmentasi")
        
        # Proses gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        test_image = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Menganalisis gambar...'):
            time.sleep(1)
            binary_image, test_signature, centroid = process_image_fully(test_image)
        
        # Tampilkan perbandingan gambar
        sub_col_1, sub_col_2 = st.columns(2)
        with sub_col_1:
            st.image(test_image, channels="BGR", caption="Citra Asli", use_column_width=True)
        
        if binary_image is not None and test_signature is not None:
            with sub_col_2:
                binary_with_centroid = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
                cv2.circle(binary_with_centroid, centroid, 10, (0, 0, 255), -1)
                st.image(binary_with_centroid, caption="Citra Hasil Segmentasi", use_column_width=True)
        else:
            st.error("Gagal mendeteksi objek pada gambar. Coba gunakan gambar lain.")

# --- Seksi Kiri (Hasil Analisis) ---
with col_right:
    st.header("Hasil Analisis")
    st.markdown("---")
    
    # Hanya tampilkan hasil jika gambar sudah diunggah dan diproses
    if test_signature is not None:
        # Klasifikasi
        best_match, min_dist, all_distances = classify_image(test_signature, ref_signatures)
        
        st.subheader("Klasifikasi Objek")
        st.metric(label="Objek Teridentifikasi Sebagai", value=best_match)
        st.metric(label="Nilai Minimum Distance", value=f"{min_dist:.4f}", help="Semakin kecil nilainya, semakin mirip objeknya.")

        st.subheader("Plot Signature")
        # Membuat plot signature dengan Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=test_signature, mode='lines', name='Signature Citra Uji', line=dict(color='#007bff')))
        fig.update_layout(
            title="Plot Angle Distance Signature (Ternormalisasi)",
            xaxis_title="Sudut (0-359 Derajat)",
            yaxis_title="Jarak Ternormalisasi (0-1)",
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white',
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

        with st.expander("Lihat Rincian Perbandingan Distance"):
            distance_data = [{"Objek Referensi": name, "Distance": f"{dist:.4f}"} for name, dist in all_distances.items()]
            st.dataframe(distance_data, use_container_width=True)
    else:
        st.info("Hasil analisis akan ditampilkan di sini setelah gambar diunggah dan diproses.")