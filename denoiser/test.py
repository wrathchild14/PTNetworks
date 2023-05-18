import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from denoiser import DenoisingDiffusionNetwork
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenoisingDiffusionNetwork(3, 3, 10).to(device)
model.load_state_dict(torch.load("trained_diffusion_model.pth"))
model.eval()

test_image = ToTensor()(Image.open("blurry.jpg")).unsqueeze(0).to(device)

denoised_image = model(test_image)
save_image(denoised_image, "denoised_image.jpg")
