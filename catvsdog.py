"""
猫狗识别命令行测试脚本

从本地验证集随机抽取一张图片，使用 ResNet50 模型预测并弹窗显示结果。
适用于不启动 Web 服务、直接在终端/脚本环境下快速验证模型效果。
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

# 固定工作目录，保证无论从哪个路径运行都能找到模型和数据集
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 自动选择计算设备：有 CUDA 则用 GPU，否则用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 验证集图像预处理（须与训练时保持一致，否则预测结果会偏差）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ResNet50 标准输入尺寸
    transforms.ToTensor(),           # 像素值归一化到 [0, 1]
    transforms.Normalize(            # ImageNet 标准化参数
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# 加载验证集：ImageFolder 按子文件夹名自动标注类别（cats / dogs）
# 注意：目录名须为 test_set，子结构为 test_set/cats/ 和 test_set/dogs/
val_dir = 'test_set'
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# ── 模型构建与权重加载 ──────────────────────────────────────────
# 使用 ResNet50 骨干网络，不加载 ImageNet 预训练（权重由下方 .pth 提供）
model = torchvision.models.resnet50(weights=None)
# 替换最后的全连接层：1000 类 → 2 类（猫 / 狗）
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_resnet50_cat_dog.pth', map_location=device))
model = model.to(device)
model.eval()

# ── 随机抽取一张图片进行预测 ────────────────────────────────────
print("=== 随机测试一张图片 ===")
image_idx = random.randint(0, len(val_dataset) - 1)
image, label = val_dataset[image_idx]       # image: 预处理后的 Tensor；label: 真实类别索引
image_batch = image.unsqueeze(0).to(device)  # 增加 batch 维度 (1, C, H, W)

with torch.no_grad():
    output = model(image_batch)
    _, predicted_idx = torch.max(output, 1)  # 取概率最大的类别索引

# ── 反归一化：将 Tensor 还原为可显示的 RGB 图像 ─────────────────
# 训练时做了 Normalize，显示前需要逆变换回 [0, 1] 范围
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img = image.cpu().numpy().transpose((1, 2, 0))  # CHW → HWC
img = std * img + mean
img = np.clip(img, 0, 1)

# ── 可视化：弹窗展示图片及预测/真实标签 ─────────────────────────
class_names = val_dataset.classes  # 如 ['cats', 'dogs']，顺序由文件夹名字母序决定
predicted_class = class_names[predicted_idx.item()]
true_class = class_names[label]

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted: {predicted_class} | True: {true_class}", fontsize=14)
plt.tight_layout()
plt.show()

# 终端同步输出，便于脚本化调用时获取结果
print(f"Predicted class: {predicted_class}")
print(f"True class: {true_class}")
