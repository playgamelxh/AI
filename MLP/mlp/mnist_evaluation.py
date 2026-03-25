import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt


# 1. 定义与训练时相同的模型结构
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


# 2. 加载训练好的模型
model = MLP()
model.load_state_dict(torch.load('mnist_mlp_model.pth'))  # 加载保存的模型参数
model.eval()  # 切换到评估模式

# 3. 准备测试数据集（用于计算准确率）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# 4. 在测试集上计算准确率
def calculate_accuracy():
    correct = 0
    total = 0

    # 关闭梯度计算，提高效率
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # 获取预测结果（概率最高的类别）
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    print(f'正确识别: {correct}/{total}')
    return accuracy


# 5. 单张图片识别函数
def predict_single_image(image_path):
    # 图片预处理（与训练时保持一致）
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整为28x28
        transforms.Grayscale(),  # 转为灰度图
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])

    # 加载并处理图片
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度

    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = output[0]
        predicted_class = torch.argmax(probabilities).item()

    # 显示图片和预测结果
    plt.imshow(image.convert('L'), cmap='gray')  # 显示灰度图
    plt.title(f'预测结果: {predicted_class}')
    plt.axis('off')
    plt.show()

    # 打印各数字的概率
    print("各数字概率分布:")
    for i in range(10):
        print(f'数字 {i}: {probabilities[i].item():.4f} ({probabilities[i].item() * 100:.2f}%)')

    return predicted_class


# 6. 执行评估和预测
if __name__ == "__main__":
    # 计算测试集准确率
    calculate_accuracy()

    # 识别单张图片（替换为你的图片路径）
    # 注意：图片应为手写数字，背景白色，数字黑色，效果最佳
    predict_single_image("test_digit.png")
