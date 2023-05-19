import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn.functional import mse_loss
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets import DenoiserDataset
from denoiser.utils import SimdLoss
from models import DenoiserAutoEncoder, DenoiserDiffusion


def train(selected_device, network, num_epochs, learning_rate=1e-4, load_prev=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    clean_dir = "./data/clean"
    noisy_dir = "./data/noisy"
    dataset = DenoiserDataset(clean_dir, noisy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # criterion = torch.nn.MSELoss()
    criterion = SimdLoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Load previous network weights if specified
    if load_prev:
        network.load_state_dict(torch.load("trained_diffusion_model.pth"))
        print("loaded previous trained model")

    for epoch in range(num_epochs):
        for batch, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(selected_device)
            clean_images = clean_images.to(selected_device)

            denoised_images = network(noisy_images)
            loss = criterion(denoised_images, clean_images)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch + 1}/{len(dataloader)}], Loss: {loss.item()}")

    torch.save(network.state_dict(), "trained_diffusion_model.pth")
    "saved model"


def test(selected_device, network, load_prev=False):
    if load_prev:
        network.load_state_dict(torch.load("trained_diffusion_model.pth"))
        print("loaded model for testing")
    network.eval()

    test_image = Image.open("pt_blurry.jpg")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_image_transformed = transform(test_image)

    test_image_tensor = test_image_transformed.unsqueeze(0).to(selected_device)

    denoised_image = network(test_image_tensor)

    print("Denoised image shape:", denoised_image.shape)

    denoised_image_np = denoised_image.squeeze(0).cpu().detach().numpy()

    # From [-1, 1] to [0, 255]
    denoised_image_np = ((denoised_image_np + 1) * 0.5 * 255).astype(np.uint8)

    import matplotlib.pyplot as plt
    plt.imshow(denoised_image_np.transpose(1, 2, 0))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 3
    out_channels = 3

    # model = DenoiserAutoEncoder(in_channels, out_channels).to(device)
    model = DenoiserDiffusion(in_channels, out_channels).to(device)

    train(device, model, num_epochs=10, load_prev=True)
    test(device, model, load_prev=False)
