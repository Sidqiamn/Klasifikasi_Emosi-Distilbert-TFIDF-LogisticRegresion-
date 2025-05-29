Model Hybrid : Distilbert + TF-IDF + Logistic Regresion

Notebook : https://colab.research.google.com/drive/1EEdWxWp3cGRp8aDVFLC0cQJBg2WzzIsm?usp=sharing


# Laporan Proyek Machine Learning - Klasifikasi Emosi
# Domain Proyek
Di era digital 2025, pemahaman emosi manusia melalui teks menjadi elemen kunci dalam aplikasi seperti kesehatan mental berbasis AI, chatbots layanan pelanggan, dan analisis sentimen di platform seperti X. Dengan melonjaknya penggunaan bahasa Indonesia di aplikasi berbasis teks, sistem otomatis untuk mendeteksi emosi sangat diperlukan untuk memberikan respons empatik dan mendukung pengambilan keputusan berbasis data. Menurut Poria et al. (2020), analisis emosi berbasis teks dapat meningkatkan interaksi manusia-mesin dengan menangkap nuansa emosional, namun tantangan utama adalah menangani keragaman bahasa Indonesia yang kaya akan konteks budaya, seperti frasa idiomatik (“hati kecilku”) atau slang digital yang terus berkembang. Acheampong et al. (2021) menyoroti bahwa deteksi emosi berbasis teks menghadapi kendala dalam menangani bahasa non-Inggris, termasuk bahasa Indonesia, yang membutuhkan model dengan kemampuan generalisasi tinggi untuk menangkap variasi linguistik lokal.

Proyek ini bertujuan mengembangkan sistem klasifikasi emosi berbasis teks dalam bahasa Indonesia untuk mengidentifikasi empat kelas emosi: happy, sadness, fear, dan stress. Dataset dikumpulkan dari sumber publik seperti happy_db, train.txt, val.txt, test.txt, stress_dataset.txt, dan dataset_negatif.txt, serta diperkaya dengan augmentasi data manual untuk menyeimbangkan distribusi kelas. Pendekatan hybrid yang mengintegrasikan DistilBERT untuk representasi semantik dengan fitur TF-IDF dan Logistic Regression dipilih untuk meningkatkan akurasi dan generalisasi, terutama pada teks dengan nuansa budaya lokal. Sanh et al. (2020) menjelaskan bahwa DistilBERT, sebagai versi ringan dari BERT, menawarkan keseimbangan antara performa dan efisiensi, menjadikannya ideal untuk deployment pada perangkat mobile melalui format TFLite.

Masalah ini signifikan karena deteksi emosi yang akurat dapat meningkatkan pengalaman pengguna, mendukung intervensi kesehatan mental melalui aplikasi mobile, dan memperkuat analisis sentimen untuk bisnis lokal. Pendekatan hybrid mengatasi keterbatasan metode berbasis kata kunci yang kurang adaptif, dengan memadukan konteks semantik DistilBERT dan pola kata eksplisit TF-IDF. Model diuji pada teks berlatar budaya Indonesia untuk memastikan relevansi, dengan evaluasi menggunakan metrik accuracy dan F1-score untuk menjamin performa andal.

Referensi:  
Acheampong, F. A., Wenyu, C., & Nunoo-Mensah, H. (2021). Text-based emotion detection: Advances, challenges, and opportunities. Engineering Reports, 3(7), e12389.

Poria, S., Hazarika, D., Majumder, N., & Mihalcea, R. (2020). Beneath the tip of the iceberg: Current challenges and new directions in sentiment analysis research. IEEE Transactions on Affective Computing, 12(1), 108–123. https://doi.org/10.1109/TAFFC.2020.3038167  

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2020). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108. https://arxiv.org/abs/1910.01108



# Business Understanding
## Problem Statements
- Keterbatasan dalam Menangani Konteks Budaya Lokal:
Bagaimana sistem dapat mengklasifikasikan emosi dari teks dalam bahasa Indonesia yang dapat membedakan antara sedih, senang, stress, dan cemas
- Ketidakseimbangan Data Emosi:
Dengan distribusi kelas yang tidak seimbang dalam dataset awal (misalnya, kelas happy mendominasi), bagaimana sistem dapat tetap akurat dalam mengklasifikasikan kelas minoritas seperti sadness atau fear?
- Efisiensi untuk Deployment Mobile:
Bagaimana cara mengembangkan model yang akurat sekaligus ringan untuk dijalankan pada perangkat mobile dengan sumber daya terbatas, sambil mempertahankan kemampuan inferensi yang cepat dan andal?

## Goals
- Pengembangan Model Klasifikasi Emosi yang Akurat:
Membangun model yang mampu mengklasifikasikan emosi (sedih, senang, stress, dan cemas) dari teks bahasa Indonesia dengan akurasi tinggi, terutama pada teks dengan konteks budaya lokal.
- Penanganan Ketidakseimbangan Data:
Mengatasi ketidakseimbangan kelas melalui augmentasi data dan teknik seperti focal loss untuk memastikan performa yang seimbang di semua kelas.
- Optimasi untuk Deployment Mobile:
Mengembangkan model yang efisien melalui pendekatan hybrid (ringan untuk inferensi) dan konversi ke format TFLite untuk memungkinkan integrasi di aplikasi mobile.

## Solution Approach
Untuk mencapai tujuan, berikut adalah pendekatan yang diusulkan:  
- Pendekatan dengan DistilBERT dan Focal Loss:
Menggunakan model DistilBERT pre-trained (distilbert-base-multilingual-cased) yang di-fine-tune untuk klasifikasi emosi. Focal loss diterapkan untuk menangani ketidakseimbangan kelas dengan parameter alpha ([6.0, 3.0, 4.0, 0.01]) dan gamma (3.0). Performa diukur dengan metrik accuracy dan F1-score (weighted).  
- Pendekatan Hybrid dengan TF-IDF dan Logistic Regression:
Menggabungkan logits DistilBERT (representasi semantik) dengan fitur TF-IDF (pola frekuensi kata) untuk melatih model Logistic Regression. Pendekatan ini lebih ringan dan skalabel dibandingkan DistilBERT penuh, dengan parameter Logistic Regression (multi_class='multinomial', C=1.0, max_iter=1000).  
- Konversi ke TFLite untuk Deployment Mobile:
Mengonversi model DistilBERT ke format TensorFlow dan TFLite untuk memastikan efisiensi inferensi pada perangkat mobile. Tokenizer disimpan terpisah untuk mendukung pemrosesan teks.  
- Optimasi Hyperparameter:
Menyesuaikan parameter pelatihan DistilBERT (epoch=10, learning rate=3e-5, batch size=8, gradient accumulation=4, early stopping patience=5) untuk meningkatkan akurasi dan mencegah overfitting.

Semua pendekatan dievaluasi dengan metrik accuracy dan F1-score untuk memastikan performa optimal dan relevansi dengan tujuan proyek.

# Data Understanding
Proyek ini menggunakan dataset yang dikumpulkan dari berbagai sumber publik, termasuk happy_db, train.txt, val.txt, test.txt, stress_dataset.txt, dan dataset_negatif.txt. Dataset ini berisi teks dalam bahasa Indonesia dengan label emosi (happy, sadness, fear, stress). Dataset dapat diakses melalui tautan berikut:
https://github.com/Sidqiamn/Dataset_Capstone_LaskarAI

## Variabel pada Dataset
- text: Teks dalam bahasa Indonesia yang mengungkapkan emosi (contoh: "Saya sangat senang hari ini karena lulus ujian!").  
- label: Label emosi yang terkait dengan teks, dengan nilai: happy, sadness, fear, atau stress.  
- labels: Versi numerik dari label (happy=0, sadness=1, fear=2, stress=3), ditambahkan selama preprocessing. 
- text_length: Jumlah kata dalam teks, dihitung untuk analisis EDA.

Fokus utama adalah pada file seperti train.txt, val.txt, test.txt, dan dataset tambahan seperti happy_db dan stress_dataset.txt, yang berisi teks dan label untuk klasifikasi emosi.

## Visualisasi dan Exploratory Data Analysis (EDA)

### Contoh Data
Visualisasi menunjukkan lima baris pertama dataset, dengan kolom text dan label. Contoh: teks "Saya sangat senang hari ini karena lulus ujian!" memiliki label happy. Ini memberikan gambaran format data dan distribusi awal.

### Distribusi Kelas

Count plot menunjukkan distribusi kelas sebelum penyeimbangan, dengan kelas happy mendominasi (frekuensi tinggi), diikuti oleh sadness, fear, dan stress yang jauh lebih sedikit. Ketidakseimbangan ini menegaskan perlunya augmentasi dan penyeimbangan data.
![sidqiamn](https://imgur.com/UNLhunO.png)
### Distribusi Panjang Teks
Histogram menunjukkan bahwa sebagian besar teks memiliki panjang <100 kata, dengan kelas happy memiliki frekuensi tertinggi. Kelas lain (sadness, fear, stress) memiliki teks yang lebih pendek dan seragam, menyoroti kebutuhan untuk menangani variasi panjang teks.
![sidqiaman](https://imgur.com/FD63Ajw.png)

### Box Plot Panjang Teks per Kelas
Box plot menunjukkan bahwa kelas happy memiliki variasi panjang teks terbesar, dengan banyak outlier (teks >1000 kata). Kelas lain memiliki median sekitar 10–20 kata, menunjukkan teks yang relatif pendek dan seragam. Ini memengaruhi pemilihan max_length=128 untuk tokenisasi.

![Imgur](https://imgur.com/2rkpsMU.png)

### Insight dari EDA:  
- Ketidakseimbangan kelas memerlukan augmentasi dan penyeimbangan data untuk mencegah bias terhadap kelas happy.  
- Variasi panjang teks, terutama pada kelas happy, menegaskan pentingnya padding dan truncation dalam tokenisasi.  
- Teks pendek mendominasi, mendukung efisiensi model DistilBERT dengan max_length=128.

# Data Preparation
Bagian ini menjelaskan langkah-langkah persiapan data untuk memastikan dataset siap digunakan dalam pemodelan klasifikasi emosi. Teknik diterapkan untuk menangani ketidakseimbangan kelas, membersihkan data, dan memformatnya untuk pelatihan.
## Pembersihan Data  
- Memeriksa dataset untuk nilai hilang (isnull) atau duplikat (duplicated). Tidak ada nilai hilang atau duplikat ditemukan.  
- Kolom tambahan seperti timestamp (jika ada) diabaikan karena tidak relevan.  
- Pembersihan memastikan integritas data dan mengurangi kompleksitas tanpa kehilangan informasi penting.
## Augmentasi Data  
- Menambahkan sampel baru secara manual untuk kelas happy (42 sampel), fear (21 sampel), sadness (12 sampel), dan stress (25 sampel) untuk meningkatkan jumlah data dan variasi teks. Contoh: "Saya sangat senang hari ini karena lulus ujian!" (happy).  
- Augmentasi memperkaya representasi kelas minoritas dan mendukung penyeimbangan data, meningkatkan generalisasi model.
## Penyeimbangan Data  
- Menggunakan undersampling untuk kelas happy (mengurangi 50% sampel) dan augmentasi untuk kelas lain.  
- Mengambil 2600 sampel per kelas, menghasilkan dataset seimbang dengan 10.400 sampel (2600 per kelas: happy, sadness, fear, stress).  
- Penyeimbangan mencegah bias terhadap kelas mayoritas dan meningkatkan performa pada kelas minoritas.

## Konversi Label ke Numerik  
- Mengonversi label teks (happy, sadness, fear, stress) ke numerik (0, 1, 2, 3) menggunakan label_map.  
- Label numerik diperlukan untuk pelatihan model machine learning seperti DistilBERT.
## Pembagian Data  
- Dataset dibagi menjadi 80% pelatihan, 10% validasi, dan 10% pengujian menggunakan train_test_split dari Hugging Face.  
- Pembagian memungkinkan evaluasi model pada data yang belum terlihat, mencegah overfitting.
## Tokenisasi  
- Menggunakan DistilBertTokenizer dengan max_length=128, padding, dan truncation untuk mengonversi teks menjadi input_ids, dan attention_mask.  
- Dataset dikonfigurasi ke format PyTorch dengan kolom input_ids, attention_mask, labels, dan text.  
- Tokenisasi menstandarikan input untuk DistilBERT dan mendukung pemrosesan batch.

# Modeling
Bagian ini membahas pengembangan model klasifikasi emosi menggunakan dua pendekatan: DistilBERT Murni dan Model Hybrid. Keduanya dirancang untuk mengklasifikasikan teks ke dalam empat kelas emosi, dengan fokus pada akurasi dan efisiensi.

## DistilBERT Murni

DistilBERT (distilbert-base-multilingual-cased) adalah model pre-trained Transformer yang di-fine-tune untuk klasifikasi emosi. Model ini memanfaatkan representasi semantik untuk menangkap konteks teks.
### Tahapan:  
- Memuat model DistilBERT dengan 4 label (happy, sadness, fear, stress).  
- Menggunakan CustomTrainer dengan focal loss untuk menangani ketidakseimbangan kelas (alpha=[6.0, 3.0, 4.0, 0.01], gamma=3.0).  
- Parameter pelatihan:  
- Epoch: 10  
- Batch size: 8 (gradient accumulation=4)  
- Learning rate: 3e-5 (cosine scheduler)  
- Warmup steps: 200  
        - Weight decay: 0.1  
        - Early stopping: Patience 5 (berdasarkan F1-score)  
        - Mixed precision (FP16) jika GPU tersedia.
    - Melatih model pada train_dataset dan mengevaluasi pada val_dataset.
### Hasil:  
- Akurasi**: 97.2% (step 850).  
- F1-score: 0.97 (weighted).  
- Contoh log pelatihan:  
    - Step 50: Accuracy=48.36%, F1-score=39.60%, Validation loss=25%.  
    - Step 850: Accuracy=97.21%, F1-score=97.22%, Validation loss=0.13.

![Imgur](https://imgur.com/Du3yCa9.png)

Model menunjukkan peningkatan signifikan, dengan training loss rendah (0.2896) dan stabilisasi performa.
### Kelebihan:  
- Menangkap konteks semantik yang kaya, termasuk frasa budaya lokal.  
- Akurasi tinggi dan generalisasi baik pada data validasi.
### Kekurangan: 
- Komputasi lebih berat, memerlukan sumber daya untuk inferensi.  
- Focal loss memerlukan penyesuaian parameter yang cermat.

## Model Hybrid (DistilBERT + TF-IDF + Logistic Regression)
Model hybrid menggabungkan logits DistilBERT dan fitur TF-IDF untuk melatih Logistic Regression, memberikan alternatif ringan untuk inferensi.
### Tahapan:  
- Ekstraksi logits: Menggunakan extract_logits untuk menghasilkan logits dari DistilBERT untuk setiap teks (bentuk: (n_samples, 4)).  
- Ekstraksi fitur TF-IDF: Menggunakan TfidfVectorizer(max_features=5000) untuk menghasilkan fitur frekuensi kata (bentuk: (n_samples, 5000)).  
- Menggabungkan fitur: Menggabungkan TF-IDF dan logits menjadi X_combined (bentuk: (n_samples, 5004)).  
- Melatih Logistic Regression: Menggunakan LogisticRegression(multi_class='multinomial', C=1.0, max_iter=1000) pada X_train_combined dan y_train.  
- Prediksi: Fungsi predict_hybrid menggabungkan pipeline untuk inferensi teks baru.
### Hasil:  
- Accuracy: 97% (test set).  
- F1-score: 0.97 (weighted).  
- Contoh prediksi pada teks uji:  
    - "Saya sangat senang hari ini karena lulus ujian!": happy  
    - "Saya pusing karena pekerjaan menumpuk.": stress  
    - "Hati kecilku berkata akan baik-baik saja.": happy

### Kelebihan:  
- Lebih ringan untuk inferensi dibandingkan DistilBERT penuh.  
- Menggabungkan kekuatan semantik (DistilBERT) dan pola kata (TF-IDF).  
- Skalabel dan efektif untuk teks dengan variasi bahasa.
### Kekurangan:  
- Bergantung pada kualitas fitur TF-IDF, yang mungkin kurang menangkap konteks kompleks.  
- Pemrosesan logits satu per satu kurang efisien untuk batch besar.

## Evaluasi
Bagian ini mengevaluasi performa model DistilBERT Murni dan Model Hybrid pada dataset pengujian menggunakan metrik accuracy, F1-score (weighted), dan confusion matrix. Hasil dianalisis untuk menilai keberhasilan model dalam mencapai tujuan proyek.
### Classification Report (Validation - Hybrid Model)

| Label     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| happy     | 0.97      | 0.96   | 0.97     | 228     |
| sadness   | 0.97      | 0.97   | 0.97     | 279     |
| fear      | 0.97      | 0.98   | 0.98     | 277     |
| stress    | 1.00      | 1.00   | 1.00     | 256     |

**Accuracy:** 0.98 (Total: 1040)  
**Macro Avg:** Precision 0.98 | Recall 0.98 | F1-score 0.98  
**Weighted Avg:** Precision 0.98 | Recall 0.98 | F1-score 0.98  

---

### Classification Report (Test - Hybrid Model)

| Label     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| happy     | 0.99      | 0.96   | 0.97     | 267     |
| sadness   | 0.95      | 0.96   | 0.96     | 256     |
| fear      | 0.96      | 0.98   | 0.97     | 266     |
| stress    | 1.00      | 1.00   | 1.00     | 251     |

**Accuracy:** 0.97 (Total: 1040)  
**Macro Avg:** Precision 0.97 | Recall 0.97 | F1-score 0.97  
**Weighted Avg:** Precision 0.97 | Recall 0.97 | F1-score 0.97  

![Imgur](https://imgur.com/T8WuSNE.png)
### Classification Report (DistilBERT Murni)

| Label     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| happy     | 0.98      | 0.94   | 0.96     | 267     |
| sadness   | 0.95      | 0.96   | 0.95     | 256     |
| fear      | 0.93      | 0.98   | 0.96     | 266     |
| stress    | 1.00      | 0.96   | 0.98     | 251     |

**Accuracy:** 0.96 (Total: 1040)  
**Macro Avg:** Precision 0.96 | Recall 0.96 | F1-score 0.96  
**Weighted Avg:** Precision 0.96 | Recall 0.96 | F1-score 0.96  

![Imgur](https://imgur.com/zCp8KOP.png)

### Interpretasi Evaluasi 
1. Model Hybrid: Mengungguli DistilBERT murni dengan akurasi 97% (vs. 96%) dan F1-score 0.97 (vs. 0.96). Performa lebih konsisten, terutama pada kelas happy (recall 0.98) dan stress (precision/recall 1.00).  

2. DistilBERT Murni: Tetap akurat, tetapi sedikit lebih banyak kesalahan pada happy (15 vs. 11) dan fear. Focal loss membantu, tetapi kombinasi TF-IDF memberikan keunggulan hybrid.  

3. Confusion Matrix: Model hybrid membuat lebih sedikit kesalahan, dengan performa hampir sempurna pada stress. Kesalahan minor pada happy dan fear mungkin karena overlap konteks (misalnya, "optimis" bisa happy atau fear).

# Inferensi dan Pengujian
## Tahapan:  
- Model Hybrid: Menggunakan predict_hybrid untuk memprediksi emosi dari teks uji. Contoh:  
    - "Saya sangat senang hari ini karena lulus ujian!": happy  
    - "Saya pusing karena pekerjaan menumpuk.": stress
- Model TFLite: Menggunakan interpreter TFLite untuk inferensi, menghasilkan logits dan prediksi kelas.
## Hasil:  
- Model hybrid dan TFLite memberikan prediksi akurat pada test_texts, dengan model hybrid sedikit lebih konsisten (97% vs. 96%).  
- Contoh: "Hati kecilku berkata akan baik-baik saja." diprediksi sebagai happy oleh kedua model, menunjukkan kemampuan menangani konteks lokal.
## Insight:  
- Model hybrid lebih efisien untuk inferensi cepat, sementara TFLite mendukung deployment mobile.  
- Kedua model menangani teks dengan konteks budaya lokal dengan baik.

# Kesimpulan Keseluruhan
- Performa: Model hybrid mencapai akurasi 97% dan F1-score 0.97, sedikit mengungguli DistilBERT murni (96%, F1-score 0.96). Kedua model efektif untuk klasifikasi emosi, dengan hybrid lebih stabil pada kelas happy dan stress.  
- Penanganan Konteks Lokal: Kedua model berhasil menangkap frasa budaya, memenuhi tujuan menangani bahasa Indonesia.  
- Efisiensi: Model hybrid lebih ringan untuk inferensi dan  TFLite memungkinkan deployment mobile dengan performa konsisten.  
- Metrik: Accuracy dan F1-score selaras dengan tujuan proyek, memastikan prediksi mendekati emosi sebenarnya.  
- Relevansi: Pendekatan hybrid meningkatkan generalisasi dibandingkan pendekatan berbasis kata kunci, mendukung aplikasi dunia nyata seperti kesehatan mental atau analisis sentimen.







