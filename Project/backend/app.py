# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import joblib
import pandas as pd
import os
import json
import traceback
import numpy as np

# -----------------------------------------
# Flask setup
# -----------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:8080"]}})

# -----------------------------------------
# Paths & constants
# -----------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_price_model.pkl")
CSV_PATH = os.path.join(MODEL_DIR, "merged_articles_transactions.csv")

# Canonical fallback classes
PRODUCT_CLASSES = ["T-shirt", "Dress", "Shirt", "Blouse", "Sweater",
                   "Jacket", "Trousers", "Shorts", "Skirt", "Vest Top"]
COLOR_CLASSES = ["Black", "White", "Red", "Blue", "Navy",
                 "Grey", "Beige", "Pink", "Green", "Brown"]

HEAD_HIDDEN_PRODUCT = 512
HEAD_HIDDEN_COLOR = 256
HEAD_DROPOUT = 0.5

# -----------------------------------------
# Model definition
# -----------------------------------------
class ResNetMultiHead(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=False,
                 num_product_classes=10, num_color_classes=10,
                 hidden_product=HEAD_HIDDEN_PRODUCT, hidden_color=HEAD_HIDDEN_COLOR,
                 dropout=HEAD_DROPOUT):
        super().__init__()
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Only resnet50 is supported here")

        self.product_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_product),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_product, num_product_classes)
        )

        self.color_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_color),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_color, num_color_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.product_head(feat), self.color_head(feat)

# -----------------------------------------
# Load PyTorch model
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Missing model file: {BEST_MODEL_PATH}")

checkpoint_raw = torch.load(BEST_MODEL_PATH, map_location=device)

if isinstance(checkpoint_raw, dict) and "model_state_dict" in checkpoint_raw:
    checkpoint_state_dict = checkpoint_raw["model_state_dict"]
    ck_product_classes = checkpoint_raw.get("product_classes", PRODUCT_CLASSES)
    ck_color_classes = checkpoint_raw.get("color_classes", COLOR_CLASSES)
else:
    checkpoint_state_dict = checkpoint_raw
    ck_product_classes = PRODUCT_CLASSES
    ck_color_classes = COLOR_CLASSES

PRODUCT_CLASSES = ck_product_classes
COLOR_CLASSES = ck_color_classes

model = ResNetMultiHead(
    num_product_classes=len(PRODUCT_CLASSES),
    num_color_classes=len(COLOR_CLASSES),
    hidden_product=HEAD_HIDDEN_PRODUCT,
    hidden_color=HEAD_HIDDEN_COLOR,
    dropout=HEAD_DROPOUT
)

def strip_module_prefix(sd):
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("module.", "")] = v
    return new_sd

try:
    model.load_state_dict(checkpoint_state_dict)
except RuntimeError:
    model.load_state_dict(strip_module_prefix(checkpoint_state_dict))

model.to(device)
model.eval()
print(f"✅ Loaded CNN model: {BEST_MODEL_PATH} on {device}")

# -----------------------------------------
# Load XGBoost model
# -----------------------------------------
if not os.path.exists(XGB_MODEL_PATH):
    raise FileNotFoundError(f"Missing XGBoost model: {XGB_MODEL_PATH}")

xgb_model = joblib.load(XGB_MODEL_PATH)
print("✅ Loaded XGBoost model:", XGB_MODEL_PATH)

# -----------------------------------------
# Load CSV reference
# -----------------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing CSV reference: {CSV_PATH}")

df_ref = pd.read_csv(CSV_PATH)
print("✅ Loaded CSV with", len(df_ref), "rows")

categorical_columns = [
    "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "perceived_colour_value_name", "perceived_colour_master_name",
    "department_name", "index_group_name", "section_name", "garment_group_name"
]
for col in categorical_columns:
    if col not in df_ref.columns:
        df_ref[col] = ""

df_encoded = pd.get_dummies(df_ref, columns=categorical_columns, drop_first=True)
feature_columns = [col for col in df_encoded.columns if col != "price"]

# -----------------------------------------
# Transforms
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------
# Helper functions
# -----------------------------------------
def preprocess_tabular(data_json):
    df = pd.DataFrame([data_json])
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = ""
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

def to_json_safe(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj

# -----------------------------------------
# Routes
# -----------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask backend with PyTorch CNN is running."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # --- CNN inference ---
        image = request.files["file"]
        img = Image.open(io.BytesIO(image.read())).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out_p, out_c = model(x)
            probs_p = torch.softmax(out_p, dim=1)
            probs_c = torch.softmax(out_c, dim=1)
            pred_prod_idx = int(torch.argmax(out_p, dim=1).item())
            pred_color_idx = int(torch.argmax(out_c, dim=1).item())

        predicted_product = PRODUCT_CLASSES[pred_prod_idx] if pred_prod_idx < len(PRODUCT_CLASSES) else "unknown"
        predicted_color = COLOR_CLASSES[pred_color_idx] if pred_color_idx < len(COLOR_CLASSES) else "unknown"

        # --- Metadata ---
        metadata_str = request.form.get("metadata")
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
            except Exception:
                metadata = eval(metadata_str)
        else:
            metadata = {}

        metadata_defaults = {
            "product_type_name": predicted_product,
            "product_group_name": "Garments",
            "graphical_appearance_name": "Solid",
            "colour_group_name": predicted_color,
            "perceived_colour_value_name": "Medium colour",
            "perceived_colour_master_name": predicted_color,
            "department_name": "Ladieswear",
            "index_group_name": "Ladieswear",
            "section_name": "Topwear",
            "garment_group_name": "Jersey Basic"
        }
        metadata = {**metadata_defaults, **metadata}

        # --- Preprocess for XGB ---
        df_input = pd.DataFrame([metadata])
        for col in categorical_columns:
            if col not in df_input.columns:
                df_input[col] = ""
        df_input = pd.get_dummies(df_input, columns=categorical_columns, drop_first=True)

        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[feature_columns]

        # --- Predict Price ---
        predicted_price = None
        try:
            pred_value = xgb_model.predict(df_input)
            predicted_price = float(pred_value[0]) * 1_000_000
        except Exception as ex:
            print("⚠️ XGBoost prediction failed:", ex)
            predicted_price = None

        # --- CSV Fallback average ---
        if predicted_price is None or np.isnan(predicted_price):
            print(f"🔄 Using CSV fallback for {predicted_product}")
            try:
                avg_price = df_ref[df_ref["product_type_name"].str.lower() == predicted_product.lower()]["price"].mean()
                if pd.notnull(avg_price):
                    predicted_price = float(avg_price) * 1_000_000
                else:
                    predicted_price = 0.0
            except Exception as csv_ex:
                print("⚠️ CSV fallback failed:", csv_ex)
                predicted_price = 0.0

        # --- Build response ---
        probs_p_list = to_json_safe(probs_p[0])
        probs_c_list = to_json_safe(probs_c[0])
        topk_p = sorted(enumerate(probs_p_list), key=lambda x: x[1], reverse=True)[:5]
        topk_c = sorted(enumerate(probs_c_list), key=lambda x: x[1], reverse=True)[:5]

        return jsonify({
            "predicted_product": predicted_product,
            "predicted_color": predicted_color,
            "predicted_price": predicted_price,
            "product_probs": probs_p_list,
            "color_probs": probs_c_list,
            "topk_product": [
                {"index": int(i), "class": PRODUCT_CLASSES[int(i)], "score": float(s)} for i, s in topk_p
            ],
            "topk_color": [
                {"index": int(i), "class": COLOR_CLASSES[int(i)], "score": float(s)} for i, s in topk_c
            ]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -----------------------------------------
# Main entry
# -----------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
