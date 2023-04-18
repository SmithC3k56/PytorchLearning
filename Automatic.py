import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


#Computing Gradients -tính toán đạo hàm của hàm mất mát (loss function) theo các tham số của mô hình là một bước quan trọng để cập nhật các trọng số (weights) của mô hình thông qua thuật toán tối ưu hóa.
loss.backward()
print(w.grad)
print(b.grad)

#Disabling Gradient Tracking
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
#another way for disabling gradient tracking with detach()
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

#tính đạo hàm của đầu vào inp đối với hàm mất mát out
inp = torch.eye(4, 5, requires_grad=True) # requires_grad for tracking gradient
out = (inp+1).pow(2).t() # tăng inp lên 1 và bình phương và chuyển vị(hoán đổi một cột thành một hàng)
out.backward(torch.ones_like(out), retain_graph=True)#backward() được gọi trên out với đối số là một tensor bằng một. Điều này tương đương với việc tính đạo hàm của out với mỗi thành phần của inp.
#retain_graph=True cho phép tính toán đạo hàm lần thứ hai trên cùng đồ thị tính toán mà không phải tạo ra đồ thị mới.
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

# Tensor Gradients and Jacobian Products
#Tensor gradients là đạo hàm của một hàm số đối với một Tensor. Khi tính toán đạo hàm của một hàm số đối với một Tensor trong PyTorch, kết quả sẽ trả về một Tensor có cùng kích thước với Tensor đầu vào, chứa các giá trị gradient tương ứng.
#Jacobian Products là tích giữa ma trận Jacobian và một Tensor đầu vào. Jacobian của một hàm số là ma trận chứa đạo hàm riêng của từng biến số trong hàm số đó. Khi tính toán Jacobian Products cho một hàm số trong PyTorch, kết quả sẽ trả về một Tensor có cùng kích thước với Tensor đầu vào, chứa các giá trị gradient tương ứng.
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

