import numpy as np
import pytorch_ssim as pytorch_ssim
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn.functional import mse_loss
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets import DenoiserDataset
from denoiser.utils import SimdLoss
from models import UNet, EncDec

import pytorch_ssim


def train(selected_device, network, num_epochs, learning_rate=1e-4, load_prev=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    clean_dir = "./data/clean"
    noisy_dir = "./data/noisy"
    dataset = DenoiserDataset(clean_dir, noisy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # criterion = torch.nn.MSELoss()
    # criterion = SimdLoss()
    criterion = torch.nn.L1Loss()
    # criterion = pytorch_ssim.SSIM()
    # criterion = pytorch_msssim.SSIM();

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # scheduler for reducing the lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    if load_prev:
        network.load_state_dict(torch.load("trained_model.pth"))
        print("loaded previous trained model")

    for epoch in range(num_epochs):
        for batch, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(selected_device)
            clean_images = clean_images.to(selected_device)

            de_noised_images = network(noisy_images)
            loss = criterion(de_noised_images, clean_images)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            scheduler.step(loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch + 1}/{len(dataloader)}], Loss: {loss.item()}")

    torch.save(network.state_dict(), "trained_model.pth")
    print("saved model")


def test(selected_device, network, load_prev=False, image_path="pt_blurry.jpg"):
    if load_prev:
        network.load_state_dict(torch.load("trained_model.pth"))
        print("loaded model for testing")
    network.eval()

    # transform image
    test_image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_image_transformed = transform(test_image)
    test_image_tensor = test_image_transformed.unsqueeze(0).to(selected_device)

    # revert back to rgb
    de_noised_image = network(test_image_tensor)
    print("Denoised image shape:", de_noised_image.shape)
    de_noised_image_np = de_noised_image.squeeze(0).cpu().detach().numpy()
    # From [-1, 1] to [0, 255]
    de_noised_image_np = ((de_noised_image_np + 1) * 0.5 * 255).astype(np.uint8)

    import matplotlib.pyplot as plt
    plt.imshow(de_noised_image_np.transpose(1, 2, 0))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 3
    out_channels = 3

    # model = EncDec(in_channels, out_channels).to(device)
    model = UNet(in_channels, out_channels).to(device)

    train(device, model, load_prev=False, num_epochs=3)
    test(device, model, load_prev=False, image_path="test_image.jpg")
