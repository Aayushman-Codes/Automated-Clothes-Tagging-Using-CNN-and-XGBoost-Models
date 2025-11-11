# Fashion Product Automated-Tagging System

An **AI-powered fashion product classification and price prediction system** that automatically tags fashion product images by **color**, **product type**, and predicts their **estimated price** using integrated **deep learning (ResNet50 CNN)** and **XGBoost** models.

---

## Project Description

This project integrates computer vision and machine learning to help e-commerce platforms automatically classify and price fashion items.

- **CNN (ResNet50)** — Classifies product **color** and **type**
- **XGBoost Regressor** — Predicts **price** from tabular product metadata
- **Flask Backend** — Serves prediction APIs
- **React Frontend** — Uploads images, displays predictions, and interacts with Flask backend

---

## Architecture Overview

| Component | Framework | Description |
|------------|------------|-------------|
| **Frontend** | React (Vite + TypeScript) | User interface for uploading images and viewing results |
| **Backend** | Flask (Python) | REST API for image & metadata inference |
| **Model 1** | ResNet50 (Keras/TensorFlow) | Deep CNN for product & color classification |
| **Model 2** | XGBoost | Predicts product price based on metadata |
| **Database** | None | Stateless system (no external DB required) |

---

## System Requirements

| Tool | Version |
|------|----------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm / yarn | Latest |
| pip | Latest |
| OS | Windows / macOS / Linux |

---

### Installation & Setup Guide

## 1) Clone the Repository
```bash
git clone https://github.com/Aayushman-Codes/Automated-Clothes-Tagging-for-E-commerce.git
cd Automated-Product-Tagging-for-E-commerce
```
## 2) Backend Setup

```bash
cd backend
```

# Create and Activate Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Model Setup Instructions (for `best_model.h5`)

### Option 1 — Auto-Download (Recommended)

1. Open `fashion_model.py`
2. Replace `YOUR_FILE_ID_HERE` with your Google Drive file ID
3. Run the following command:
   ```bash
   python fashion_model.py
   ```
4. The script will:
- Check if best_model.h5 exists
- Automatically download it from Google Drive (if missing)
- Save it in backend/models/best_model.h5

#### How to Get Your Google Drive File ID:
- Upload best_model.h5 to Google Drive
- Right-click → Share → Anyone with link
- Copy the link: https://drive.google.com/file/d/FILE_ID_HERE/view
- Extract the FILE_ID_HERE portion and place it inside fashion_model.py

### Option 2 — Manual Setup

1. **Download** `best_model.h5` manually from your own Google Drive or shared link.  
2. **Place the file inside: backend/models/best_model.h5**
3.  **Verify your folder structure:**
backend/
├── app.py
├── models/
│ ├── best_model.h5
│ ├── xgb_price_model.pkl
│ └── merged_articles_transactions.csv

---


# Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

# Run Backend Server
```bash
python app.py
```

Backend runs at: http://localhost:5000


## 3) Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:8080

## 4) Integrating Models

Place the following files in the backend/models/ folder:

backend/

├── models/

│   ├── best_model.h5

│   ├── xgb_price_model.pkl

│   └── merged_articles_transactions.csv





# Model Information — Fashion Product Auto-Tagging (ResNet50 + XGBoost)

This project integrates **deep learning (ResNet50 CNN)** and **XGBoost regression** to classify and predict properties of fashion products for e-commerce automation.

---

## CNN — Fashion Product Classifier (ResNet50)

### Framework
- **TensorFlow/Keras 2.20.0**
- **Architecture:** ResNet50 (Transfer Learning)
- **Input Size:** 224×224×3
- **Output:** 20 total classes (10 colors + 10 product types)

### Model Parameters
| Metric | Value |
|--------|--------|
| Total Parameters | 24,783,508 |
| Trainable Parameters | 1,190,676 (4.8%) |
| Frozen Parameters | 23,592,832 (95.2%) |
| Model Size | 96 MB |

---

## Classes

### Colors (10)
Black, White, Red, Blue, Navy, Grey, Beige, Pink, Green, Brown

### Products (10)
T-shirt, Dress, Shirt, Blouse, Sweater, Jacket, Trousers, Shorts, Skirt, Vest Top

---

## Training Details
| Hyperparameter | Value |
|----------------|--------|
| **Loss Function** | Binary Cross-Entropy |
| **Optimizer** | Adam |
| **Regularization** | Dropout (0.5, 0.3) + Batch Normalization |
| **Pretraining** | ImageNet |
| **Transfer Learning** | ResNet50 base (frozen layers) + custom dense layers |

---




### Project Structure

Automated-Product-Tagging-for-E-commerce/

├── backend/

│   ├── app.py

│   ├── models/

│   │   ├── best_model.h5

│   │   ├── xgb_price_model.pkl

│   │   └── merged_articles_transactions.csv

│   ├── requirements.txt

│   └── .env

├── frontend/

│   ├── src/

│   ├── vite.config.ts

│   ├── package.json

│   └── tsconfig.json

└── README.md




