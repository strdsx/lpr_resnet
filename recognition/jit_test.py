import torch
import torchvision
from torchvision import transforms, datasets
import os
from PIL import Image
import numpy as np
import sys
import argparse
import resnet

def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('Run ',device)

	model = resnet.ResNet(resnet.Bottleneck, [3,8,36,3])
	model.load_state_dict(torch.load('./model/resnet152_best-72.ckpt'))

	traced_script_module = torch.jit.trace(model, torch.rand(1,3,32,32).unsqueeze(0))
	traced_script_module.save('script_model.pt')


if __name__ == '__main__':
	main()
