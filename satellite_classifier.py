# satellite_classifier.py
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# 1. Use pretrained ResNet (transfer learning = faster)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes

# 2. Simple data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])

# 3. Load data
from torchvision.datasets import ImageFolder
dataset = ImageFolder('data/', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Quick training (just 5 epochs for demo)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# 5. Save model
torch.save(model.state_dict(), 'satellite_model.pth')