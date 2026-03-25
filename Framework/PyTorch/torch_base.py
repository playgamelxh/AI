import torch
import numpy as np

#  create tensors directly from data
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(x_data)

# create from a numpy data
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# create from other tensor
x_ones = torch.ones_like(x_data)
print(x_ones) # sampe shape

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor, ones_tensor, zeros_tensor)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on:{tensor.device}")
else:
    print(f"Device cuda not available")

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print("tensor:",tensor)

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")