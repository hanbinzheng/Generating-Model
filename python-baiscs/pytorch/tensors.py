import torch
import math
import numpy as np

'''
https://docs.pytorch.org/tutorials/beginner/basics/tensor_tutorial.html
'''

# directly from the data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"From the data directly: \n{x_data}\n")

# from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"From a numpy arrya: \n{x_np}\n")

# 
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n{x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n{x_rand} \n")

# shape is a tuple of tensor dimensions
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor: \n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}\n")

# Tensor qttributes describe their shape, datatype, and the device
tensor = torch.rand(3, 4)

print(f"Shape of the tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
    tensor = tensor.to('cuad')
else:
    print("\nStupid zhb, you don't have a gpu!")

data_0 = [[0.00, 0.01, 0.02, 0.03], [0.10, 0.11, 0.12, 0.13], [0.20, 0.21, 0.22, 0.23], [0.30, 0.31, 0.32, 0.33]]
data_1 = [[1.00, 1.01, 1.02, 1.03], [1.10, 1.11, 1.12, 1.13], [1.20, 1.21, 1.22, 0.23], [1.30, 1.31, 1.32, 1.33]]
data_2 = [[2.00, 2.01, 2.02, 2.03], [2.10, 2.11, 2.12, 2.13], [2.20, 2.21, 2.22, 2.23], [2.30, 2.31, 2.32, 2.33]]
tensor = torch.tensor(data_0)
tensor_1 = torch.tensor(data_1)
tensor_2 = torch.tensor(data_2)
print(tensor)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
print()

t1 = torch.cat([tensor, tensor_1, tensor_2], dim=1)
print(t1)
print()
print()

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
x1 = tensor @ tensor

print(x1)
print(y1)
print(z1)


# tensor1 @ tensor2: m * n @ n * p
# tensor1 @ tensor2.T m * n @ k * n
# tensor1 * tensor2 m * n  * n * k

# tensor1.mul(tensor2) == tensor1 * tensor2
# tensor1.matmul(tensor2) == tensor1 @ tensor2

agg = tensor.sum()
print(agg)
agg_item = agg.item()
print(agg_item, type(agg_item))
