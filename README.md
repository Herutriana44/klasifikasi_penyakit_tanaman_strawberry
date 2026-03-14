# Klasifikasi Penyakit Tanaman Strawberry dengan GoogleNet

Proyek ini menggunakan arsitektur **GoogleNet (GoogLeNet)** untuk klasifikasi gambar penyakit tanaman strawberry.

## Struktur Project

```
klasifikasi_penyakit_tanaman_strawberry/
├── train_googlenet.ipynb    # Notebook training (Colab/Kaggle)
├── run_flask_ngrok.ipynb    # Notebook deploy web dengan ngrok (Colab)
├── app.py                   # Flask web app
├── templates/
│   └── index.html
├── models/                  # Tempat model hasil training
│   ├── googlenet_model.pth
│   └── class_names.json
├── requirements.txt
└── README.md
```

## 1. Training Model (Google Colab / Kaggle)

1. Buka `train_googlenet.ipynb` di Colab atau Kaggle
2. Siapkan dataset dengan struktur:
   ```
   dataset/
   ├── train/
   │   ├── class1/   (gambar)
   │   ├── class2/
   │   └── ...
   └── val/
       ├── class1/
       ├── class2/
       └── ...
   ```
3. Upload dataset (zip untuk Colab, atau Add Dataset untuk Kaggle)
4. Aktifkan GPU: Runtime > Change runtime type > GPU (Colab) atau Settings > Accelerator (Kaggle)
5. Jalankan semua cell
6. Download `googlenet_model.pth` dan `class_names.json` dari output

## 2. Web Flask (Lokal)

1. Copy `googlenet_model.pth` dan `class_names.json` ke folder `models/`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan:
   ```bash
   python app.py
   ```
4. Buka http://localhost:5000

## 3. Deploy di Google Colab dengan Ngrok

1. Zip folder project ini
2. Buka `run_flask_ngrok.ipynb` di Google Colab
3. Upload zip project
4. Upload model dan class_names (jika belum ada di zip)
5. Jalankan semua cell
6. Buka URL ngrok yang muncul untuk mengakses web dari internet

**Ngrok:** Daftar gratis di [ngrok.com](https://ngrok.com) untuk auth token. Uncomment dan isi `ngrok.set_auth_token('...')` di cell 4.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.0+
- Pillow
