import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from common import calculate_accuracy

model_path = 'fashion_minist_mlp_model.pth'

# 1. 数据加载（MNIST 数据集）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量（0~1 归一化）
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（均值、标准差）
])
train_data = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 2. 定义 MLP 模型（输入层784 → 隐藏层128 → 隐藏层64 → 输出层10）
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),  # 输入层：展平 28×28 图像为 784 维向量
            nn.Linear(784, 128),  # 隐藏层1：784→128（全连接）
            nn.ReLU(),  # 激活函数
            nn.Linear(128, 64),   # 隐藏层2：128→64
            nn.ReLU(),
            nn.Linear(64, 10),    # 输出层：64→10（10分类）
            nn.Softmax(dim=1)     # 激活函数（输出概率）
        )

    def forward(self, x):
        return self.layers(x)  # 前向传播

# 3. 初始化模型、损失函数、优化器
model = MLP()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适配 Softmax 输出）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

if not os.path.exists(model_path):
    # 4. 训练模型（10轮迭代）
    max_loop = 30
    for epoch in range(max_loop):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播 + 参数更新
            optimizer.zero_grad()  # 清空梯度（避免累积）
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新参数

            running_loss += loss.item() * inputs.size(0)

        # 打印每轮损失
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{max_loop}, Loss: {epoch_loss:.4f}')

    print("训练完成！")

    # 5. 保存训练好的模型
    # 保存模型的状态字典（推荐方式）
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存为 {model_path}")

# 6. 加载模型（推理或继续训练）
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为评估模式（关闭 Dropout 等）

# 加载测试数据集
test_data = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
# 计算测试集准确率
accuracy = calculate_accuracy(model, test_loader)
