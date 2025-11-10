import torch

# 加载 .pth 文件（两种情况）
# 情况1：仅保存了模型权重（state_dict）
state_dict = torch.load("mnist_mlp_model.pth")
print("模型权重结构：")
for key, value in state_dict.items():
    print(f"层名称：{key}，参数形状：{value.shape}")

print("=================\r\n")

# 情况2：保存了整个模型（包括结构+权重）
model = torch.load("mnist_mlp_model.pth")
print("模型结构：")
print(model)  # 直接打印模型网络结构