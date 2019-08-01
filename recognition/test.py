import torch
import torchvision
from torchvision import transforms, datasets
import torch.jit
import os
from PIL import Image
import numpy as np
import sys
import argparse
import resnet


def main(root_image, root_model):
	print("\nInput image ==> ",root_image)
	print("Current model ===> ",root_model)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("Run ",device)

	## ResNet-50 : [3, 4, 6, 3]
    ## ResNet-101 : [3, 4, 23, 3]
    ## ResNet-152 : [3, 8, 36, 3]

	model = resnet.ResNet(resnet.Bottleneck, [3,8,36,3])
	model.load_state_dict(torch.load(root_model))
	model.to(device)

	# Input Image Transforms
	loader = transforms.Compose([
		transforms.Resize((32,32)),
		transforms.ToTensor(),
		transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
		# transforms.Normalize([0.5],[0.5]),
		])

	model.eval()
	with torch.no_grad():
		# label list : 67 class
		label = os.listdir("./dataset/img/")
		label.sort()
		
		# to RGB
		image = Image.open(root_image)
		image = image.convert("RGB")

		# to Tensor, Cuda
		image_tensor = loader(image).unsqueeze(0)
		image_tensor = image_tensor.to(device)
		output = model(image_tensor)

        # Predict
		_, predicted = torch.max(output.data, 1)

        # Calc Probability
		s_max = torch.nn.Softmax(dim=0)
		prob = s_max(output.squeeze(0))
		prob_np = prob.cpu().numpy()

		index = int(predicted.item())
		
		predicted_name = label[index]
		probability = round(prob_np[index] * 100, 2)

		print("Predicted ===> name : {},  probability : {}".format(predicted_name, probability))
		
		'''
		class_index = 0
		for p in prob_np:
			print(str(class_index) + " ==> " + str(p * 100) + "%")
			class_index += 1
		'''






def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--image',
		type=str,
		default='test.jpg',
		help='Image name'
		)
	parser.add_argument('--model',
		type=str,
		default='./model/resnet-1.ckpt',
		help='model path'
		)
	return parser.parse_args(argv)

def test():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	net = resnet.ResNet(resnet.Bottleneck, [3, 8, 36, 3])
	net = net.to(device)
	
	net.eval()
	with torch.no_grad():
		img_tensor = torch.randn(1,3,32,32)
		img_tensor = img_tensor.to(device)
		y = net(img_tensor)
		print(y.size())

if __name__ == '__main__':
	args = parse_arguments(sys.argv[1:])
	main(args.image, args.model)



