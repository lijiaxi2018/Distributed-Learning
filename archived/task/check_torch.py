import torch

print("cuda.is_available", torch.cuda.is_available())
print("cuda.device_count", torch.cuda.device_count())
print("cuda.current_device", torch.cuda.current_device())

print("cuda.get_device_name(0)", torch.cuda.get_device_name(0))