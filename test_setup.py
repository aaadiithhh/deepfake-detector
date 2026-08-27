import torch
import torchvision

print("Setup successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")