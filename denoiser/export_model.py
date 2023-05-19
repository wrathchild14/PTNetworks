import torch

from denoiser.models import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet(3, 3).to(device)
PATH = 'trained_unet_model.pth'
net.load_state_dict(torch.load(PATH))
net.eval()
x = torch.randn((1, 3, 400, 400)).cuda()

torch.onnx.export(net,  # model being run
                  x,  # model input
                  "model_unet.onnx",  # where to save the model
                  export_params=True,  # store the trained weights
                  opset_version=11,  # the ONNX version
                  do_constant_folding=True,
                  input_names=['input'],  # set model input names
                  output_names=['output'],  # set model output names
                  )
