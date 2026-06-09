import io
import os
import sys
import threading
import webbrowser

import torch
import torch.nn as nn
import torchvision
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision import transforms

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["cats", "dogs"]
CLASS_LABELS = {"cats": "猫", "dogs": "狗"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = None


def load_model():
    global model
    weights_path = "best_resnet50_cat_dog.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到模型权重文件: {weights_path}")

    net = torchvision.models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net = net.to(device)
    net.eval()
    model = net
    print(f"模型已加载，使用设备: {device}")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filepath>")
def static_files(filepath):
    if os.path.isfile(filepath):
        return send_from_directory(".", filepath)
    return jsonify({"error": "文件不存在"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型未加载"}), 500

    if "image" not in request.files:
        return jsonify({"error": "请上传图片"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "无法解析图片，请上传有效的 JPG/PNG 文件"}), 400

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    predicted = CLASS_NAMES[torch.argmax(probs).item()]

    return jsonify({
        "prediction": predicted,
        "label": CLASS_LABELS[predicted],
        "probabilities": probabilities,
    })


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("正在加载模型，首次启动可能需要几秒…")
    try:
        load_model()
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        sys.exit(1)

    threading.Timer(1.2, open_browser).start()
    print("服务已启动 → http://127.0.0.1:5000")
    print("按 Ctrl+C 停止服务")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
