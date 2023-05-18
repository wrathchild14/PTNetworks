import torch.optim as optim
import torch.nn.functional

from torch.nn.functional import mse_loss
from torchvision import transforms
from torch.utils.data import DataLoader
from denoiser import DenoiserEncDec, DenoiserDiffusion
from denoiser_dataset import DenoiserDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((256, 256)),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    clean_dir = "./data/clean"
    noisy_dir = "./data/noisy"
    dataset = DenoiserDataset(clean_dir, noisy_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    in_channels = 3
    out_channels = 3

    # model = DenoiserDiffusion(in_channels, out_channels).to(device)
    model = DenoiserEncDec(in_channels, out_channels).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # load prev network
    # model.load_state_dict(torch.load("trained_diffusion_model.pth"))

    num_epochs = 10
    for epoch in range(num_epochs):
        for batch, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # denoised_images = to_tensor(denoised_images)  # Convert denoised images to tensors
            # clean_images = to_tensor(clean_images)

            denoised_images = model(noisy_images)
            loss = criterion(denoised_images, clean_images)
            optimizer.zero_grad()

            mse = mse_loss(denoised_images, clean_images)
            # ssim_loss = 1 - ssim(denoised_images, clean_images, data_range=1, size_average=True)
            # total_loss = 0.5 * mse + 0.5 * ssim_loss
            # total_loss.backward()
            mse.backward()

            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch + 1}/{len(dataloader)}], Loss: {loss.item()}")

    torch.save(model.state_dict(), "trained_diffusion_model.pth")
