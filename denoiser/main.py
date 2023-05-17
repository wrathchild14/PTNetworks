import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from denoiser import DenoisingDiffusionNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ImageFolder("./data", transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = DenoisingDiffusionNetwork(3, 3).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for batch, (noisy_images, _) in enumerate(dataloader):
        noisy_images = noisy_images.to(device)

        clean_images = model(noisy_images)

        loss = criterion(clean_images, noisy_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch + 1}/{len(dataloader)}], Loss: {loss.item()}")

torch.save(model.state_dict(), "trained_diffusion_model.pth")
