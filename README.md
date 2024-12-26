# 🐝Analisis Kesehatan Sarang Lebah Dengan Menganalisis Gambar Lebah

## **Deskripsi Proyek**
Kesehatan koloni lebah merupakan faktor yang sangat penting dalam menjaga ekosistem dan produksi pangan global. Namun, koloni lebah menghadapi berbagai ancaman dari organisme pengganggu, seperti Varroa destructor (kutu varroa), belatung sarang lebah, serta masalah dengan ratu lebah yang hilang. Salah satu tantangan terbesar dalam pemeliharaan lebah adalah mengidentifikasi secara dini gejala-gejala kesehatan yang buruk pada sarang lebah.

## **Sumber Data**
Data yang digunakan dalam proyek ini diambil dari dataset [Honey Bee Annotated Images](https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images) yang tersedia di Kaggle. Dataset ini berisi gambar-gambar yang telah diberi anotasi dengan berbagai kategori kondisi kesehatan sarang lebah, yang sangat berguna untuk pelatihan model pembelajaran mesin dalam analisis gambar. Untuk mengakses dan mengunduh dataset ini, Anda dapat mengunjungi halaman Kaggle berikut: [Honey Bee Annotated Images](https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images).

## **Model yang Digunakan**

Proyek ini menggunakan dua model utama untuk analisis gambar sarang lebah:

1. **Convolutional Neural Network (CNN)**
   - CNN adalah salah satu model paling populer dalam pengolahan gambar. Model ini menggunakan lapisan konvolusi untuk mengekstrak fitur dari gambar, yang kemudian digunakan untuk melakukan klasifikasi. Dalam proyek ini, CNN digunakan untuk mengenali berbagai kondisi kesehatan sarang lebah dari gambar yang diberikan.

2. **MobileNet**
   - MobileNet adalah arsitektur jaringan saraf yang lebih ringan dan lebih cepat, yang sangat cocok untuk digunakan pada perangkat dengan sumber daya terbatas atau untuk aplikasi yang memerlukan pemrosesan gambar dalam waktu nyata.

