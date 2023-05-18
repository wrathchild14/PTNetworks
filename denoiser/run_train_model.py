import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from denoiser import DenoisingDiffusionNetworkEncDec
from denoiser_dataset import DenoiserDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # transforms.Resize((256, 256)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    clean_dir = "./data/clean"
    noisy_dir = "./data/noisy"
    dataset = DenoiserDataset(clean_dir, noisy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    in_channels = 3
    out_channels = 3

    model = DenoisingDiffusionNetworkEncDec(in_channels, out_channels).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.load_state_dict(torch.load("trained_diffusion_model.pth"))

    num_epochs = 100
    for epoch in range(num_epochs):
        for batch, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            denoised_images = model(noisy_images)

            loss = criterion(denoised_images, clean_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch + 1}/{len(dataloader)}], Loss: {loss.item()}")

    torch.save(model.state_dict(), "trained_diffusion_model.pth")
