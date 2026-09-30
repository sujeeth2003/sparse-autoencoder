'''
import torch
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
torch.set_default_dtype(dtype)
torch.set_default_device(device)

print(f"using {device} device and {dtype} dtype")

a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
c = torch.randn((), requires_grad=True)
d = torch.randn((), requires_grad=True)

