import torch

# 检查 MPS 是否支持
print(torch.backends.mps.is_available())   # 输出 True 表示支持
print(torch.backends.mps.is_built())       # 输出 True 表示 PyTorch 已编译 MPS 支持

# 使用 MPS 设备
if torch.backends.mps.is_available():
    print("MPS 设备可用")
    device = torch.device("mps")  # Apple GPU
else:
    print("MPS 设备不可用，回退到 CPU")  # 输出错误信息并使用 CPU 作为备用设备
    device = torch.device("cpu")  # 回退到 CPU

# 示例：将张量放在 MPS 设备上
x = torch.randn(2, 3, device=device)
print(x.device)  # 输出 "mps"