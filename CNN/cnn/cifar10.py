import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from common import calculate_accuracy

model_path = 'cifar10_cnn_model.pth'

# 1. 数据加载（CIFAR-10 数据集）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量（0~1 归一化）
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化（均值、标准差）
])
train_data = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 示例：简单CNN处理RGB图像
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 初始化模型、损失函数、优化器
model = CNN()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适配 Softmax 输出）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

if not os.path.exists(model_path):
    # 4. 训练模型（10轮迭代）
    max_loop = 10
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
test_data = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
# 计算测试集准确率
accuracy = calculate_accuracy(model, test_loader)
