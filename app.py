import streamlit as st
import cv2
import numpy as np
import os
import plotly.graph_objects as go
import time
import pandas as pd

# --- KODE UNTUK MENGUBAH TAMPILAN (BACKGROUND & FONT) ---
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

/* Styling untuk Expander/Dropdown */
[data-testid="stExpander"] {
    background-color: rgba(0, 0, 0, 0.2);
    border: none !important;
    box-shadow: none !important;
    border-radius: 10px;
    margin-bottom: 1rem;
}

[data-testid="stExpander"] summary {
    padding: 0.75rem 1rem;
    border-radius: 10px;
    transition: background-color 0.3s ease;
}

[data-testid="stExpander"] summary:hover {
    background-color: rgba(43, 89, 148, 0.5) !important;
}

[data-testid="stExpander"] summary p {
    color: #FFFFFF !important;
    font-weight: 600;
}

[data-testid="stExpander"] svg {
    color: #FFFFFF !important;
}

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
    page_icon="",
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
    distances = [np.sqrt((p[0][0] - cx)*2 + (p[0][1] - cy)*2) for p in main_contour]
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

col_left, col_center, col_right = st.columns([1.2, 2, 1.2])

# --- Seksi Kiri (Kontrol) ---
with col_left:
    st.header("Panel Kontrol")
    st.markdown("---")
    
    ref_signatures = load_reference_signatures("reference_images")
    
    if not ref_signatures:
        st.error("Folder reference_images tidak ditemukan atau kosong.")
    else:
        st.success(f"{len(ref_signatures)} kelas referensi berhasil dimuat.")
        with st.expander("Lihat Kelas Referensi"):
            st.table(pd.DataFrame(ref_signatures.keys(), columns=["Nama Kelas"]))

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Unggah Gambar", 
        type=["jpg", "jpeg", "png"],
        help="Pilih gambar objek untuk dianalisis."
    )

# --- Seksi Tengah (Perbandingan Citra) ---
with col_center:
    st.title("Analisis Similaritas Citra Digital")
    
    test_signature = None
    binary_image = None

    if uploaded_file is None:
        st.info("Selamat Datang! Silakan unggah sebuah gambar pada Panel Kontrol di sebelah kiri untuk memulai analisis.")
    else:
        st.header("Input & Hasil Segmentasi")
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        test_image = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Menganalisis gambar...'):
            time.sleep(1)
            binary_image, test_signature, centroid = process_image_fully(test_image)
        
        sub_col_1, sub_col_2 = st.columns(2)
        with sub_col_1:
            st.image(test_image, channels="BGR", caption="Citra Asli", use_column_width=True)
        
        if binary_image is not None and test_signature is not None:
            with sub_col_2:
                binary_with_centroid = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
                if centroid:
                    cv2.circle(binary_with_centroid, centroid, 10, (0, 0, 255), -1)
                st.image(binary_with_centroid, caption="Citra Hasil Segmentasi", use_column_width=True)
        else:
            st.error("Gagal mendeteksi objek pada gambar. Coba gunakan gambar lain.")

# --- Seksi Kanan (Hasil Analisis) ---
with col_right:
    st.header("Hasil Analisis")
    st.markdown("---")
    
    if test_signature is not None:
        best_match, min_dist, all_distances = classify_image(test_signature, ref_signatures)
        
        st.subheader("Klasifikasi Objek")
        st.metric(label="Objek Teridentifikasi Sebagai", value=best_match)
        st.metric(label="Nilai Minimum Distance", value=f"{min_dist:.4f}", help="Semakin kecil nilainya, semakin mirip objeknya.")

        st.subheader("Plot Signature")
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
            distance_df = pd.DataFrame(all_distances.items(), columns=["Objek Referensi", "Distance"])
            distance_df["Distance"] = distance_df["Distance"].map('{:.4f}'.format)
            st.dataframe(distance_df, use_container_width=True, hide_index=True)

        with st.expander("Nilai Distance Berdasarkan Sudut Theta Citra Biner"):
            num_blocks = 8
            rows_per_block = 360 // num_blocks
            p_values = np.arange(1, 361)

            # --- ✅ [PERBAIKAN BAGIAN 1] ---
            # Membuat DataFrame untuk diunduh (dengan kolom duplikat, tidak masalah untuk CSV)
            df_chunks_for_csv = []
            for i in range(num_blocks):
                start = i * rows_per_block
                end = start + rows_per_block
                dist_values_formatted = [f"{val:.5f}" for val in test_signature[start:end]]
                chunk_df = pd.DataFrame({
                    'P': p_values[start:end],
                    'Dist': dist_values_formatted
                }).reset_index(drop=True)
                df_chunks_for_csv.append(chunk_df)
            
            download_df = pd.concat(df_chunks_for_csv, axis=1)
            csv_data = download_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
               label="Unduh Data sebagai CSV",
               data=csv_data,
               file_name='nilai_distance_berdasarkan_theta.csv',
               mime='text/csv',
               help="Unduh data dalam format tabel multi-kolom."
            )
            st.markdown("---")

            # --- ✅ [PERBAIKAN BAGIAN 2] ---
            # Membuat TAMPILAN di aplikasi menggunakan st.columns untuk menghindari error
            # st.write("Pratinjau Data:") # Header opsional
            cols = st.columns(num_blocks)
            for i in range(num_blocks):
                with cols[i]:
                    start = i * rows_per_block
                    end = start + rows_per_block
                    
                    # Buat DataFrame kecil & terpisah untuk setiap kolom
                    display_chunk_df = pd.DataFrame({
                        'P': p_values[start:end],
                        'Dist': [f"{val:.5f}" for val in test_signature[start:end]]
                    })
                    
                    st.dataframe(
                        display_chunk_df,
                        hide_index=True,
                        use_container_width=True
                    )
            # --- Akhir Perbaikan ---
            
    else:
        st.info("Hasil analisis akan ditampilkan di sini setelah gambar diunggah dan diproses.")
