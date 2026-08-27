from torchvision import models, transforms
from PIL import Image
import torch

# Load a pretrained model (we'll swap this for a deepfake-specific one later)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Prepare the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("test_image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Run it through the model
with torch.no_grad():
    output = model(input_tensor)

print("Image loaded and processed successfully!")
print(f"Output shape: {output.shape}")