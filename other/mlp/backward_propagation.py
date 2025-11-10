import numpy as np

# 1. 前向传播（保存中间结果）
x = np.random.randn(784, 1)    # 输入
y = np.zeros((10, 1))
y[3] = 1.0                     # 真实标签（假设第3类）

# 网络参数初始化
W1 = np.random.randn(128, 784)
b1 = np.random.randn(128, 1)
W2 = np.random.randn(64, 128)
b2 = np.random.randn(64, 1)
W3 = np.random.randn(10, 64)
b3 = np.random.randn(10, 1)

# 前向计算
z1 = W1 @ x + b1  # @ NumPy 中的矩阵乘法运算符（等价于np.matmul()函数），用于计算两个数组的矩阵乘积
a1 = np.maximum(0, z1)         # ReLU激活
z2 = W2 @ a1 + b2
a2 = np.maximum(0, z2)
z3 = W3 @ a2 + b3
print("z3:", z3)

# 改进：数值稳定的Softmax计算
max_z3 = np.max(z3)  # 取出z3向量中的最大值
exp_z3 = np.exp(z3 - max_z3)  # 对向量每个值减去最大值，避免指数溢出 再做指数运算
y_hat = exp_z3 / np.sum(exp_z3)  # Softmax输出（无溢出）
print("y_hat:", y_hat)

# 2. 计算损失（交叉熵）
# 避免log(0)：给y_hat加一个极小值epsilon（如1e-10）
epsilon = 1e-10
L = -np.sum(y * np.log(y_hat + epsilon))  # 防止log(0)

# 3. 反向传播
# 输出层误差 delta3（交叉熵+Softmax的导数仍为y_hat - y，不受数值稳定影响）
delta3 = y_hat - y

# 输出层参数梯度
dW3 = delta3 @ a2.T
db3 = delta3

# 隐藏层2误差 delta2
delta2 = (W3.T @ delta3) * (a2 > 0)  # ReLU导数：a2>0时为1，否则0

# 隐藏层2参数梯度
dW2 = delta2 @ a1.T
db2 = delta2

# 隐藏层1误差 delta1
delta1 = (W2.T @ delta2) * (a1 > 0)

# 隐藏层1参数梯度
dW1 = delta1 @ x.T
db1 = delta1

# 4. 参数更新（学习率eta=0.01）
eta = 0.01
W1 -= eta * dW1
b1 -= eta * db1
W2 -= eta * dW2
b2 -= eta * db2
W3 -= eta * dW3
b3 -= eta * db3