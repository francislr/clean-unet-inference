import argparse
import json

import torch

from clean_unet_inference.module import CleanUNet

parser = argparse.ArgumentParser(prog='ExportCleanUNetONNX', description='Export CleanUNet to ONNX')

parser.add_argument('config', help="Model configuration file", type=argparse.FileType('r'))
parser.add_argument('checkpoint', help="Model checkpoint filename")
parser.add_argument('onnx', help="ONNX exported filename")
parser.add_argument('device', default='cpu', choices=['cpu','gpu'], help="choose target device")

args = parser.parse_args()


if args.device = 'gpu' and torch.cuda.is_available():
    device = 'cuda'
else:
    if not torch.cuda.is_available()
        print('WARN: CUDA backend in not avilable. Device set to CPU')
    device = 'cpu'


checkpoint = torch.load(args.checkpoint, map_location=device)
config = json.load(args.config)

net = CleanUNet(**config["network_config"])
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

dummy_data = torch.zeros((1, 1000))

torch.onnx.export(
    model=net.to(device),
    args=dummy_data.to(device),
    f=args.onnx,
    do_constant_folding=True,
    export_params=True,
    input_names=[
        "noisy_audio_16khz",
    ],
    output_names=[
        "clean_audio_16khz",
    ],
    dynamic_axes={
        'noisy_audio_16khz': {0: 'batch_size', 1: 'audio_data'},
        'clean_audio_16khz': {0: 'batch_size', 1: 'audio_data'},
    }
)
