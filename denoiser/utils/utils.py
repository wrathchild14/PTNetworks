import torch


class SimdLoss(torch.nn.Module):
    def __init__(self):
        super(SimdLoss, self).__init__()
        self.similarity_loss = torch.nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        cosine_sim = self.similarity_loss(x, y)
        loss = torch.mean(1 - cosine_sim)
        return loss


def evaluate(selected_device, network, transform, dataloader, criterion=torch.nn.MSELoss()):
    network.eval()
    total_loss = 0
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():
        for batch, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(selected_device)
            clean_images = clean_images.to(selected_device)
            de_noised_images = network(noisy_images)
            loss = criterion(de_noised_images, clean_images)
            mse = torch.mean((de_noised_images - clean_images) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            # ssim = torch.mean(torch.functional.image.ssim(de_noised_images, clean_images, data_range=1))

            total_loss += loss.item() * noisy_images.size(0)
            total_mse += mse.item() * noisy_images.size(0)
            total_psnr += psnr.item() * noisy_images.size(0)
            # total_ssim += ssim.item() * noisy_images.size(0)
            num_samples += noisy_images.size(0)

    avg_loss = total_loss / num_samples
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    # avg_ssim = total_ssim / num_samples

    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    # print(f"Average SSIM: {avg_ssim:.4f}")
