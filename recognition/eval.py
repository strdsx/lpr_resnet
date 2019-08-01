import torch
import torchvision
from torchvision import transforms, datasets
import os
from PIL import Image
import resnet
import argparse
import sys
from binascii import unhexlify, hexlify


def eval(root_image, root_model):
	print("\nCurrent test image path ==> ",root_image)
	print("Current model ==> ",root_model)
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# resnet50  : [3, 4, 6, 3]
	# resnet101 : [3, 4, 23, 3]
	# resnet152 : [3, 8, 36, 3]
	model = resnet.ResNet(resnet.Bottleneck, [3, 8, 36, 3])
	model.load_state_dict(torch.load(root_model))
	model.to(device)

	transforms_test = transforms.Compose([
		# transforms.Pad(4),
		# transforms.RandomHorizontalFlip(),
		# transforms.RandomCrop(10),
		transforms.Resize((32,32)),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	
	# test loader
	testset = torchvision.datasets.ImageFolder(
		root=root_image,
		transform=transforms_test)

	test_loader = torch.utils.data.DataLoader(
		testset,
		batch_size=16,
		shuffle=False,
		num_workers=4)
	
	class_correct = list(0. for i in range(67))
	class_total = list(0. for i in range(67))
	classes = os.listdir("./data/rename_headline_piap/")
	classes.sort()
	class_count = len(classes)
	
	model.eval()
	with torch.no_grad():
		total_corr = 0.
		correct = 0.
		total = 0.
		for images, labels in test_loader:
			images=images.to(device)
			labels=labels.to(device)
			outputs = model(images)
			# best percentage
			_, predicted = torch.max(outputs.data, 1)

			total += labels.size(0)
			total_corr += (predicted == labels).sum().item()
			correct = (predicted == labels).squeeze()
			
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += correct[i].item()
				class_total[label] += 1

		for i in range(class_count):
			### For euc-kr decoding
			# unhexlify(classes[i]).decode('euc-kr')
			# print('Accuracy of %s : %2d %%' %(unhexlify(classes[i]).decode('euc-kr')[0], 100*class_correct[i]/class_total[i]))
				print('Accuracy of %s ==> %2d %%' %(classes[i], 100*class_correct[i]/class_total[i]))

		print("Accuracy of the network ======> %2d %%"%(100*total_corr/total))


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--imagefolder',
		type=str,
		default="./data/rename_gray_headline_piap",
		help = "Default ==> ./data/rename_gray_headline_piap"
		)
	parser.add_argument('--model',
		type=str,
		default="./model/resnet152_best-72.ckpt",
		help="Default ==> ./model/resnet-1.ckpt"
		)
	return parser.parse_args(argv)

if __name__ == '__main__':
	args = parse_arguments(sys.argv[1:])
	eval(args.imagefolder, args.model)
