import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets import DenoiserDataset
from models import UNet

import matplotlib.pyplot as plt

def train(selected_device, network, num_epochs, learning_rate=1e-4, load_prev=False, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    clean_dir = "./denoiser/data/generated_images/clean"
    noisy_dir = "./denoiser/data/generated_images/noisy"
    dataset = DenoiserDataset(clean_dir, noisy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.L1Loss()

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # scheduler for reducing the lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    if load_prev:
        network.load_state_dict(torch.load("trained_model.pth"))
        print("loaded previous trained model")

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for batch, (noisy_images, clean_images) in pbar:
            noisy_images = noisy_images.to(selected_device)
            clean_images = clean_images.to(selected_device)

            # diffused_images = network.diffusion(noisy_images, clean_images, num_steps=1000, step_size=learning_rate)

            de_noised_images = network(noisy_images)
            loss = criterion(de_noised_images, clean_images)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            scheduler.step(loss)

            total_loss += loss.item()
            pbar.set_description(f"Loss: {total_loss/(batch+1):.4f}")

        torch.save(network.state_dict(), "trained_model.pth")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f} and checkpoint created")


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

    # clean image
    # clean = Image.open("test_image_overfitted_clean.jpg")
    # clean_image = transform(clean)
    # clean_image_tensor = clean_image.unsqueeze(0).to(selected_device)

    # revert back to rgb

    # diffusion step
    # test_image_tensor = network.diffusion(test_image_tensor, clean_image_tensor, step_size=1e-4)

    de_noised_image = network(test_image_tensor)
    print("De-noised image shape:", de_noised_image.shape)
    de_noised_image_np = de_noised_image.squeeze(0).cpu().detach().numpy()
    # From [-1, 1] to [0, 255]
    de_noised_image_np = ((de_noised_image_np + 1) * 0.5 * 255).astype(np.uint8)

    # save image
    output_image = Image.fromarray(de_noised_image_np.transpose(1, 2, 0))
    image_name = image_path.split(".")[0]
    output_image.save(f"{image_name}_denoised.jpg")

    plt.imshow(de_noised_image_np.transpose(1, 2, 0))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 2
    BATCH_SIZE = 8
    
    in_channels = 3
    out_channels = 3

    model = UNet(in_channels, out_channels).to(device)

    # train_diffusion(selected_device=device, network=model, load_prev=False, num_epochs=EPOCHS, batch_size=BATCH_SIZE)
    # train(device, model, load_prev=False, num_epochs=EPOCHS, batch_size=BATCH_SIZE)
    test(device, model, load_prev=True, image_path="test_image_overfitted.jpg")
