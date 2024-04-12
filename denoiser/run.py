import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from datasets import DenoiserDataset
from models import UNet
from diffusion import Diffusion


def train(selected_device, network, num_epochs, transform, dataloader, learning_rate=1e-4, load_prev=False, criterion=nn.L1Loss()):


    optimizer = optim.AdamW(network.parameters(), lr=learning_rate) # goated optimizer

    # scheduler for reducing the lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    diffusion = Diffusion(img_size=transform.transforms[0].size[0], device=selected_device) # get the scale from the transform

    if load_prev:
        network.load_state_dict(torch.load("../trained_model.pth"))
        print("Successfully loaded previous model for training")

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for batch, (noisy_images, clean_images) in pbar:
            noisy_images = noisy_images.to(selected_device)
            clean_images = clean_images.to(selected_device)

            t = diffusion.sample_timesteps(noisy_images.shape[0]).to(selected_device)
            noisy_images, _ = diffusion.noise_images(noisy_images, t)

            noisy_image = noisy_images[0].squeeze(0).cpu().detach().numpy()
            noisy_image = ((noisy_image + 1) * 0.5 * 255).astype(np.uint8)
            plt.imshow(noisy_image.transpose(1, 2, 0))
            plt.axis('off')
            plt.show()

            de_noised_images = network(noisy_images)
            # if epoch > num_epochs / 2:
            #     # de_noised_images = checkpoint(network, noisy_images)
            # else:
            #     diffused_images = network.diffusion(noisy_images, clean_images)
            #     de_noised_images = network(diffused_images)
                # de_noised_images = checkpoint(network, diffused_images)

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


def test(selected_device, network, transform, load_prev=False, image_path="pt_blurry.jpg", save_img=False):
    if load_prev:
        network.load_state_dict(torch.load("../trained_model.pth"))
        print("loaded model for testing")
    network.eval()

    # transform image
    test_image = Image.open(image_path)
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
    if save_img:
        output_image = Image.fromarray(de_noised_image_np.transpose(1, 2, 0))
        image_name = image_path.split(".")[0]
        output_image.save(f"{image_name}_denoised.jpg")
        
    plt.imshow(de_noised_image_np.transpose(1, 2, 0))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    argparaser = argparse.ArgumentParser()
    argparaser.add_argument("--epochs", type=int, default=5)
    argparaser.add_argument("--batch_size", type=int, default=32)
    argparaser.add_argument("--learning_rate", type=float, default=1e-4)
    argparaser.add_argument("--load_prev", type=bool, default=False)
    argparaser.add_argument("--test", type=bool, default=False)
    argparaser.add_argument("--image_path", type=str, default="../test_data/test_image3.jpg")
    argparaser.add_argument("--save_img", type=bool, default=False)
    args = argparaser.parse_args()

    criterion = torch.nn.L1Loss()
    # criterion = torch.nn.SmoothL1Loss()
    # criterion = torch.nn.MSELoss()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    LOAD_PREV = args.load_prev
    TEST = args.test
    IMAGE_PATH = args.image_path
    SAVE_IMG = args.save_img
    
    in_channels = 3
    out_channels = 3

    model = UNet(in_channels, out_channels).to(device)

    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # scale down for bigger batch
    if BATCH_SIZE >= 32:
        transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    clean_dir = 'data/generated_images/clean'
    noisy_dir = 'data/generated_images/noisy'
    dataset = DenoiserDataset(clean_dir, noisy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Init training: {EPOCHS} epochs, {BATCH_SIZE} batch size, {LEARNING_RATE} learning rate, criterion: {criterion}, transform resize: {transform.transforms[0].size}")

    train(device, network=model, transform=transform, dataloader=dataloader, load_prev=LOAD_PREV, num_epochs=EPOCHS, learning_rate=LEARNING_RATE, criterion=criterion)
    test(device, network=model, transform=transform, load_prev=False, save_img=False, image_path=IMAGE_PATH)
