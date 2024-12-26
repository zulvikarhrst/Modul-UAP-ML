import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from utils import (
    load_model,
    preprocess_image,
    predict_health_with_probabilities,
    generate_grad_cam,
    overlay_heatmap,
)

# Konfigurasi Streamlit
st.set_page_config(
    page_title="Analisis Kesehatan Sarang Lebah",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("🐝 Analisis Kesehatan Sarang Lebah")
st.subheader("Unggah gambar lebah Anda atau masukkan URL untuk analisis kesehatan sarang lebah.")

# Input Nama dan NIM
st.sidebar.title("Identitas Pengguna")
username = st.sidebar.text_input("Nama Pengguna")
nim = st.sidebar.text_input("NIM")

if username and nim:
    st.sidebar.success("Identitas lengkap! Silakan melanjutkan.")

    # Sidebar untuk pengaturan input
    st.sidebar.title("Pengaturan Input")
    upload_option = st.sidebar.radio("Metode Input", ("Unggah File", "URL"))

    # Fungsi untuk mendapatkan gambar dari URL
    def get_image_from_url(url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Pastikan URL valid

            # Validasi header Content-Type
            if "image" not in response.headers["Content-Type"]:
                raise ValueError("URL tidak mengarah ke file gambar.")

            # Buka gambar
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            raise ValueError(f"Error mengambil gambar dari URL: {e}")

    # Fungsi validasi ekstensi URL
    def is_valid_image_url(url):
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
        return any(url.lower().endswith(ext) for ext in valid_extensions)

    # Input gambar melalui URL atau file
    image = None
    if upload_option == "URL":
        image_url = st.sidebar.text_input("Masukkan URL Gambar")
        if image_url:
            if not is_valid_image_url(image_url):
                st.error("URL tidak memiliki ekstensi gambar yang valid.")
            else:
                try:
                    image = get_image_from_url(image_url)
                    st.image(image, caption="Gambar dari URL", use_column_width=True)
                except Exception as e:
                    st.error(f"{e}")
    elif upload_option == "Unggah File":
        uploaded_file = st.file_uploader("Unggah gambar lebah (format: .jpg, .png)", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Sidebar untuk pengaturan analisis
    st.sidebar.title("Pengaturan Analisis")
    model_option = st.sidebar.selectbox(
        "Pilih model analisis:",
        ("MobileNet", "CNN")
    )

    # Jika gambar valid telah dimuat
    if image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Gambar yang diproses", use_column_width=True)

        with col2:
            # Preprocessing gambar
            processed_image = preprocess_image(image, target_size=(224, 224))

            # Memuat model dan prediksi
            model = load_model(model_option)
            class_probabilities = predict_health_with_probabilities(model, processed_image)

            # Visualisasi hasil analisis
            st.write("### Hasil Analisis:")
            for class_name, prob in class_probabilities.items():
                st.write(f"{class_name}: {prob:.6f}")

            # Prediksi tertinggi
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            confidence = class_probabilities[predicted_class]
            st.success(f"Prediksi: **{predicted_class}**")
            st.info(f"Tingkat Kepercayaan: **{confidence:.6f}**")

            # Grad-CAM visualisasi
            st.write("### Grad-CAM Visualisasi:")
            heatmap = generate_grad_cam(
                model,
                processed_image,
                class_index=list(class_probabilities.keys()).index(predicted_class),
                layer_name="conv_pw_13_relu",
            )
            overlayed_image = overlay_heatmap(heatmap, image)
            st.image(overlayed_image, caption="Grad-CAM Overlay", use_column_width=True)

            # Visualisasi probabilitas dengan bar chart
            st.write("### Visualisasi Probabilitas:")
            class_names = list(class_probabilities.keys())
            class_probs = list(class_probabilities.values())
            fig, ax = plt.subplots()
            ax.barh(class_names, class_probs, color="skyblue")
            ax.set_xlabel("Probabilities")
            ax.set_title("Class Probabilities")
            st.pyplot(fig)
    else:
        st.warning("Silakan unggah file atau masukkan URL gambar untuk melanjutkan analisis.")
else:
    st.sidebar.warning("Harap masukkan nama dan NIM Anda untuk melanjutkan.")
