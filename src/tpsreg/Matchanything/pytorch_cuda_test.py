import torch
print(torch.cuda.is_available())  # Should return True if CUDA is enabled
print(torch.cuda.get_device_name(0))  # Should return your GPU's name
