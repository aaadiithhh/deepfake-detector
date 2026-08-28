import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load the model architecture (same as training)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load your trained weights
model.load_state_dict(torch.load("deepfake_detector.pth", map_location="cpu"))
model.eval()

# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Classes in the order ImageFolder assigned them during training
classes = ["fake", "real"]

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    print(f"\n--- {image_path} ---")
    print(f"Prediction: {classes[predicted_class]}")
    print(f"Confidence: {probabilities[predicted_class].item() * 100:.2f}%")
    print(f"Full breakdown -> fake: {probabilities[0].item()*100:.2f}%, real: {probabilities[1].item()*100:.2f}%")

predict("test_real.jpg")
predict("test_fake.jpg")