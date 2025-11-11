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
| **Model 1** | ResNet50 (Pytorch) | Deep CNN for product & color classification |
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
Create venv
```bash
python -m venv venv
```
Activate venv

On Windows:
```bash
venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```

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

│   ├── best_model.pth

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

│   │   ├── best_model.pth

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




