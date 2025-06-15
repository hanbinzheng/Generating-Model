# PyTorch Tensor 基础用法与操作汇总
# 本文件整理并标注了你今日用到的所有 PyTorch 知识点，包括张量创建、属性、索引、变换、拼接与切分、广播、矩阵乘法、以及统计函数等。

import torch
import numpy as np

# ======== Tensor 的创建方式 ========
# 从 Python 数据直接创建
x_data = torch.tensor([[1, 2], [3, 4]])

# 从 numpy 数组创建（共享内存）
np_array = np.array([[1, 2], [3, 4]])
x_np = torch.from_numpy(np_array)

# 从另一个 tensor 派生（拷贝 shape 或 dtype）
x_ones = torch.ones_like(x_data)                     # 创建全 1，shape 相同
torch.rand_like(x_data, dtype=torch.float)          # 随机值填充，指定 dtype

# ======== Tensor 的属性 ========
torch.tensor(7).shape                               # 0-D tensor
torch.tensor([1, 2, 3]).shape                       # 1-D
torch.tensor([[1], [2], [3]]).shape                 # 2-D
torch.rand(4, 3, 2).shape                           # 高维

# tensor 的设备与类型
x = torch.tensor([1])
x = x.to('cuda')                                    # 转到 GPU
x.device, x.dtype

# 快速创建函数（指定 shape 和类型）
torch.rand((2, 3))
torch.ones((2, 3), dtype=torch.int)
torch.zeros((2, 3), dtype=torch.float)

# ======== 索引与切片 ========
tensor = torch.tensor([[0.00, 0.01, 0.02, 0.03],
                       [0.10, 0.11, 0.12, 0.13],
                       [0.20, 0.21, 0.22, 0.23],
                       [0.30, 0.31, 0.32, 0.33]])
tensor[0]               # 第一行
tensor[:, 0]            # 第一列
tensor[..., -1]         # 最后一列

# 多维索引
x = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                  [[7, 8], [9, 10], [11, 12]]])
x[:, 1, 0]              # 跨 batch 索引
x[1, :, 1]              # 指定 batch 的多个值

# 高维索引示例（4D）
x = torch.arange(3*4*5*6).reshape(3, 4, 5, 6)
x[0, :, 1, 4]           # 访问具体元素
x[:, :, :, 1]           # 所有的第 4 维 index=1
x.permute(1, 0, 2, 3)   # 维度置换
x.transpose(0, 1)       # 简化版 transpose

# ======== Tensor 变换与操作 ========
a = torch.ones(2, 3)
b = torch.zeros(2, 3)
torch.cat([a, b], dim=0)          # 拼接，纵向
torch.cat([a, b], dim=1)          # 拼接，横向
torch.stack([a, b], dim=0)        # 新增一维

torch.split(torch.arange(10), 2)             # 等分切割
torch.split(torch.arange(10), [3, 4, 3])      # 指定长度切割

x = torch.arange(6).reshape(2, 3)
x.reshape(-1, 2)                              # 自动推理另一维

# squeeze / unsqueeze 去除或添加维度
y = torch.zeros(1, 2, 1, 3)
y.squeeze()                                  # 去除所有为 1 的维度
y.squeeze(0)                                 # 去除指定维度
z = torch.tensor([1, 2, 3])
z.unsqueeze(0)                               # shape: (1, 3)
z.unsqueeze(1)                               # shape: (3, 1)

# 综合练习：unsqueeze + stack + reshape + split
x1 = torch.randn(2, 3).unsqueeze(0)           # shape: (1, 2, 3)
x2 = torch.randn(2, 3).unsqueeze(0)
stacked = torch.stack([x1, x2], dim=0)       # shape: (2, 1, 2, 3)
reshaped = stacked.reshape(2, 6)             # flatten 后重组
parts = torch.split(reshaped, 3, dim=1)      # 切分

# stack 不同 dim 的含义
x = torch.arange(2*3*4).reshape(2, 3, 4)
torch.stack([x[0], x[1]], dim=0)             # shape: (2, 3, 4)
torch.stack([x[0], x[1]], dim=1)             # shape: (3, 2, 4)

# split 沿 dim=1 分割 3 个 tensor
parts = torch.split(x, 1, dim=1)

# ======== 张量的算术运算（逐元素） ========
x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.tensor([[10., 20.], [30., 40.]])
x + y
x - y
x * y
x / y
x ** 2

# ======== 广播机制（Broadcasting） ========
a = torch.tensor([[1.], [2.], [3.]])         # shape: (3,1)
b = torch.tensor([10., 20.])                # shape: (2,) -> broadcast
broadcast_sum = a + b                       # 输出 shape: (3,2)

# ======== 矩阵乘法（Matmul） ========
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
A @ B

# batch 矩阵乘法（bmm）
X = torch.randn(3, 2, 4)
Y = torch.randn(3, 4, 5)
Z = torch.bmm(X, Y)                          # shape: (3, 2, 5)

# 错误示例：维度不匹配报错
try:
    torch.mm(torch.randn(2, 3), torch.randn(4, 2))
except RuntimeError as e:
    print("RuntimeError:", e)

# ======== 统计函数：sum / mean 等 ========
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
x.sum()                                      # 所有元素和
x.sum(dim=0)                                 # 每列求和
x.sum(dim=1, keepdim=True)                   # 每行求和并保留维度
x.mean(), x.mean(dim=1)

