import torch
from torchvision import transforms
from torchvision.utils import save_image
from denoiser import DenoiserEncDec, DenoiserDiffusion
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenoiserEncDec(3, 3).to(device)
# model = DenoiserDiffusion(3, 3).to(device)
model.load_state_dict(torch.load("trained_diffusion_model.pth"))
model.eval()

test_image = Image.open("blurry.jpg")
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
test_image = transform(test_image).unsqueeze(0).to(device)

print(test_image.shape)
print(torch.min(test_image), torch.max(test_image))


denoised_image = model(test_image)
denoised_image = denoised_image.clamp(0.0, 1.0)  # Clamp values to [0, 1]
save_image(denoised_image, "denoised_image.jpg")
