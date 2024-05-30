from train import test
from models import UNet
from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(3, 3).to(device)
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
IMAGE_PATH = "test_data/test_image_random.jpg"

test(device, network=model, transform=transform, load_prev=True, save_img=False, image_path=IMAGE_PATH)
