"""
Coffee Disease Detection - Flask Backend
=========================================
File structure:
    coffee_app/
    ├── app.py                  ← this file
    ├── best_coffee_cnn.pth     ← your trained model (copy here)
    ├── templates/
    │   └── index.html          ← frontend file
    └── uploads/                ← auto-created, stores temp images

Install:
    pip install flask torch torchvision pillow

Run:
    python app.py
Then open: http://localhost:5000
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MODEL_PATH  = "best_coffee_cnn.pth"
CLASS_NAMES = ["Healthy", "Miner", "Phoma", "Rust"]
IMG_SIZE    = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disease info shown on the website
DISEASE_INFO = {
    "Healthy": {
        "description": "The leaf appears healthy with no signs of disease.",
        "severity": "None",
        "treatment": "No treatment needed. Continue regular care and monitoring.",
        "color": "#22c55e"
    },
    "Miner": {
        "description": "Leaf Miner damage — tiny larvae tunnel inside the leaf creating visible trails.",
        "severity": "Moderate",
        "treatment": "Use neem oil spray or appropriate insecticide. Remove heavily affected leaves.",
        "color": "#f59e0b"
    },
    "Phoma": {
        "description": "Phoma Leaf Spot — a fungal disease causing dark brown spots on leaves.",
        "severity": "Moderate to High",
        "treatment": "Apply copper-based fungicide. Improve air circulation and avoid overhead watering.",
        "color": "#ef4444"
    },
    "Rust": {
        "description": "Coffee Leaf Rust — most serious coffee disease, causing orange powdery spots.",
        "severity": "High",
        "treatment": "Apply fungicide immediately. Remove infected leaves. Consider resistant varieties.",
        "color": "#dc2626"
    }
}

# ─────────────────────────────────────────────
# MODEL DEFINITION (must match training code)
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )
    def forward(self, x):
        return self.block(x)


class CoffeeCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
model = CoffeeCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    results = [
        {"class": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 2)}
        for i in range(len(CLASS_NAMES))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    top = results[0]
    return top["class"], top["confidence"], results


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        predicted_class, confidence, all_probs = predict(filepath)
        info = DISEASE_INFO[predicted_class]
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "description": info["description"],
            "severity": info["severity"],
            "treatment": info["treatment"],
            "color": info["color"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
