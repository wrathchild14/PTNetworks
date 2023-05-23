import torch

from denoiser.models import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet(3, 3).to(device)
PATH = 'trained_model.pth'
net.load_state_dict(torch.load(PATH))
net.eval()
x = torch.randn((1, 3, 400, 400)).cuda()

torch.onnx.export(net, x, "exported_model.onnx", export_params=True, opset_version=11, do_constant_folding=True,
                  input_names=['input'], output_names=['output'], )
