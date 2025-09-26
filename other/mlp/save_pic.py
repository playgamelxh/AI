import torch
from torchvision import datasets, transforms
from PIL import Image

# 加载MNIST测试集（已下载过则不会重复下载）
test_data = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.ToTensor()
)

# 选择测试集中的第1张图片（索引0，可改为0-9999之间任意数）
img_tensor, label = test_data[0]  # img_tensor形状：[1,28,28]，label是真实数字

# 将张量转为PIL图片（还原为0-255的灰度图）
img = transforms.ToPILImage()(img_tensor)

# 保存为PNG文件
img.save('test_digit.png')
print(f"已保存MNIST测试集图片（真实数字：{label}）为 test_digit.png")