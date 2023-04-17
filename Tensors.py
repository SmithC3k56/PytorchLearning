import torch
import numpy as np


# Create a tensors from numpy array
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# Directly from data
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#With random or constant values

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

#Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
#Arithmetic operations: torch.add(), torch.sub(), torch.mul(), torch.div(), torch.pow(), torch.remainder(), torch.floor_divide() and other
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T # Cách viết tắt của nhân matrix
y2 = tensor.matmul(tensor.T) # Cách viết tắt của nhân matrix

y3 = torch.rand_like(y1) #Để tạo ra một tensor mới với kích thước giống như kết quả tính toán
torch.matmul(tensor, tensor.T, out=y3) # Việc sử dụng tham số out trong các hàm này giúp tiết kiệm bộ nhớ bằng cách sử dụng cùng một vùng nhớ để lưu kết quả tính toán.


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#Single-element tensors -tensor Chỉ chứa một thành phần duy nhất
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#In-place operations: các phép toán trên tensor mà không tạo ra một tensor mới, mà thay đổi trực tiếp giá trị của tensor ban đầu.
print(f"{tensor} \n")
tensor.add_(5)#phương thức add_ thay đổi giá trị của x mà không tạo ra một tensor mới. Nếu sử dụng phương thức add (không có hậu tố _), nó sẽ trả về một tensor mới chứa kết quả, và x không bị thay đổi.
print(tensor)


#Convert
#Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# Thay đổi tensor
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")