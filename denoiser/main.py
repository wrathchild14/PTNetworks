import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from denoiser import DenoisingDiffusionNetwork
from denoiser_dataset import DenoiserDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
clean_dir = "./data/clean"
noisy_dir = "./data/noisy"
dataset = DenoiserDataset(clean_dir, noisy_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


in_channels = 3
out_channels = 3
num_steps = 10

model = DenoisingDiffusionNetwork(in_channels, out_channels, num_steps).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100
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
