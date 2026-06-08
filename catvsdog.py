import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import os
import random
import matplotlib.pyplot as plt
import numpy as np

# 固定工作目录，保证能找到文件
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理（和训练时保持一致）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载验证集
val_dir = 'test_set1'
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# 重建模型 + 加载你之前的最佳权重
model = torchvision.models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_resnet50_cat_dog.pth', map_location=device))
model = model.to(device)
model.eval()

# 随机测试一张图片
print("=== 随机测试一张图片 ===")
image_idx = random.randint(0, len(val_dataset)-1)
image, label = val_dataset[image_idx]
image_batch = image.unsqueeze(0).to(device)

# 预测
with torch.no_grad():
    output = model(image_batch)
    _, predicted_idx = torch.max(output, 1)

# 反归一化并显示图片（带文字）
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img = image.cpu().numpy().transpose((1, 2, 0))
img = std * img + mean
img = np.clip(img, 0, 1)

# 图片和文字一起显示
class_names = val_dataset.classes
predicted_class = class_names[predicted_idx.item()]
true_class = class_names[label]

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')
plt.title(f"Predicted: {predicted_class} | True: {true_class}", fontsize=14)
plt.tight_layout()
plt.show()

# 终端同步输出
print(f"Predicted class: {predicted_class}")
print(f"True class: {true_class}")