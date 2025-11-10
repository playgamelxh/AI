#前向传播
import numpy as np

# 1. 定义输入（模拟一张MNIST图像，784维）
x = np.random.randn(784, 1)  # 随机生成784×1的输入向量# 2. 定义网络参数（权重和偏置，形状需匹配层与层的维度）
print("x:", x)
W1 = np.random.randn(128, 784)  # 第1层权重：128×784
print("W1:", W1)
b1 = np.random.randn(128, 1)    # 第1层偏置：128×1
print("b1:", b1)
W2 = np.random.randn(64, 128)   # 第2层权重：64×128
print("W2:", W2)
b2 = np.random.randn(64, 1)     # 第2层偏置：64×1
print("b2:", b2)
W3 = np.random.randn(10, 64)    # 输出层权重：10×64
print("W3:", W3)
b3 = np.random.randn(10, 1)     # 输出层偏置：10×1# 3. 前向传播计算# 第1隐藏层
print("b3:", b3)
z1 = np.dot(W1, x) + b1         # 线性变换：128×1 = (128×784)·(784×1) + (128×1)
print("z1:", z1)
a1 = np.maximum(0, z1)          # ReLU激活：128×1# 第2隐藏层  用于对两个数组（或标量）进行元素级的最大值比较，返回一个新数组，其中每个元素是对应位置x和y中的最大值。
print("a1:", a1)
z2 = np.dot(W2, a1) + b2        # 线性变换：64×1 = (64×128)·(128×1) + (64×1)
print("z2:", z2)
a2 = np.maximum(0, z2)          # ReLU激活：64×1# 输出层
print("a2:", a2)
z3 = np.dot(W3, a2) + b3        # 线性变换：10×1 = (10×64)·(64×1) + (10×1)
print("z3:", z3)
y_hat = z3                      # 输出预测值：10×1（若分类可加Softmax）print("输入形状：", x.shape)print("第1隐藏层输出形状：", a1.shape)print("第2隐藏层输出形状：", a2.shape)print("输出层预测形状：", y_hat.shape)  # 应输出 (10, 1)