import numpy as np
import torch.optim as optim
import torch.nn.functional

from PIL import Image
from torch.nn.functional import mse_loss
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import DenoiserEncDec, DenoiserDiffusion
from datasets import DenoiserDataset


def train(selected_device, network, num_epochs, learning_rate=0.0001, load_prev=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    clean_dir = "./data/clean"
    noisy_dir = "./data/noisy"
    dataset = DenoiserDataset(clean_dir, noisy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # load prev network
    if load_prev:
        network.load_state_dict(torch.load("trained_diffusion_model.pth"))

    for epoch in range(num_epochs):
        for batch, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(selected_device)
            clean_images = clean_images.to(selected_device)

            denoised_images = network(noisy_images)
            loss = criterion(denoised_images, clean_images)
            optimizer.zero_grad()

            mse = mse_loss(denoised_images, clean_images)
            mse.backward()

            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch + 1}/{len(dataloader)}], Loss: {loss.item()}")

    torch.save(network.state_dict(), "trained_diffusion_model.pth")


def test(selected_device, network, load_prev=False):
    if load_prev:
        network.load_state_dict(torch.load("trained_diffusion_model.pth"))
    network.eval()

    test_image = Image.open("blurry.jpg")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_image_transformed = transform(test_image)

    test_image_tensor = test_image_transformed.unsqueeze(0).to(selected_device)

    denoised_image = network(test_image_tensor)

    print("Denoised image shape:", denoised_image.shape)

    denoised_image_np = denoised_image.squeeze(0).cpu().detach().numpy()

    # from [-1, 1] to [0, 255]
    denoised_image_np = ((denoised_image_np + 1) * 0.5 * 255).astype(np.uint8)
    # denoised_image_pil = Image.fromarray(denoised_image_np.transpose(1, 2, 0))
    # denoised_image_pil.save("denoised_image.jpg")
    import matplotlib.pyplot as plt

    plt.imshow(denoised_image_np.transpose(1, 2, 0))
    plt.axis('off')
    plt.show()

    # save_image(denoised_image, "denoised_image.jpg")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 3
    out_channels = 3

    # model = DenoiserDiffusion(in_channels, out_channels).to(device)
    model = DenoiserEncDec(in_channels, out_channels).to(device)

    train(device, model, 100)
    # saves the image locally
    test(device, model)
