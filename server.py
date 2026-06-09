"""
猫狗识别 Web 服务

基于 Flask 提供 HTTP 接口，加载 ResNet50 模型对上传图片进行猫/狗二分类。
启动方式：python server.py  或双击 start.bat
访问地址：http://127.0.0.1:5000
"""

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

# 将工作目录切换到脚本所在位置，确保能正确找到模型权重和静态文件
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# 优先使用 GPU，不可用时自动回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别索引与中文标签的映射（顺序须与训练时 ImageFolder 的 classes 一致）
CLASS_NAMES = ["cats", "dogs"]
CLASS_LABELS = {"cats": "猫", "dogs": "狗"}

# 图像预处理流水线，参数与训练/验证阶段保持一致
# Resize(224)  → ResNet50 标准输入尺寸
# Normalize      → ImageNet 预训练模型的均值与标准差
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 全局模型实例，在 load_model() 中初始化，避免每次请求重复加载
model = None


def load_model():
    """加载 ResNet50 结构并读取本地训练好的权重文件。"""
    global model
    weights_path = "best_resnet50_cat_dog.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到模型权重文件: {weights_path}")

    # 构建 ResNet50，不加载 ImageNet 预训练权重（已由本地 .pth 覆盖）
    net = torchvision.models.resnet50(weights=None)
    # 将原 1000 类全连接层替换为 2 类输出（猫 / 狗）
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net = net.to(device)
    net.eval()  # 推理模式：关闭 Dropout，BatchNorm 使用运行均值
    model = net
    print(f"模型已加载，使用设备: {device}")


@app.route("/")
def index():
    """返回前端页面 index.html。"""
    return send_from_directory(".", "index.html")


@app.route("/<path:filepath>")
def static_files(filepath):
    """提供示例图片等静态资源（test_set、newtest_set 等）。"""
    if os.path.isfile(filepath):
        return send_from_directory(".", filepath)
    return jsonify({"error": "文件不存在"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    """
    接收前端上传的图片，执行模型推理并返回 JSON 结果。

    请求：multipart/form-data，字段名 image
    响应：{ prediction, label, probabilities }
    """
    if model is None:
        return jsonify({"error": "模型未加载"}), 500

    if "image" not in request.files:
        return jsonify({"error": "请上传图片"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400

    # 将上传的字节流解码为 RGB 图像（统一三通道，避免 RGBA/灰度图引发维度错误）
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "无法解析图片，请上传有效的 JPG/PNG 文件"}), 400

    # 预处理 → 增加 batch 维度 (1, 3, 224, 224) → 送入设备
    tensor = transform(image).unsqueeze(0).to(device)

    # 关闭梯度计算以节省内存、加速推理
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]  # 将 logits 转为概率分布

    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    predicted = CLASS_NAMES[torch.argmax(probs).item()]

    return jsonify({
        "prediction": predicted,       # 英文类别名，如 "cats"
        "label": CLASS_LABELS[predicted],  # 中文标签，如 "猫"
        "probabilities": probabilities,    # 各类别置信度
    })


def open_browser():
    """服务启动后自动打开默认浏览器。"""
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("正在加载模型，首次启动可能需要几秒…")
    try:
        load_model()
    except Exception as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        sys.exit(1)

    # 延迟 1.2 秒再打开浏览器，等待 Flask 完成绑定端口
    threading.Timer(1.2, open_browser).start()
    print("服务已启动 → http://127.0.0.1:5000")
    print("按 Ctrl+C 停止服务")
    # use_reloader=False 防止调试模式下模型被加载两次
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
