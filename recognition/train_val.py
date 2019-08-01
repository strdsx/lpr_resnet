# Function : Train with Validation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import os
import resnet as Res
import time
import copy
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import random


def train_val(is_inception=False):
	### Set Trainset
	transforms_train = transforms.Compose([
		transforms.Resize((32,32)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		])

	trainset = torchvision.datasets.ImageFolder(
		root='dataset/img',
		transform=transforms_train,
		)
	
	### Set Validation
	transforms_valid = transforms.Compose([
		transforms.Resize((32,32)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		])

	validset = torchvision.datasets.ImageFolder(
		root='dataset/img',
		transform=transforms_valid,
		)
	

	# train : valid = 7 : 3
	valid_size = 0.3
	num_train = len(trainset)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	
	shuffle = True
	random_seed = 500
	if shuffle:
		np.random.seed(random_seed)
		np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	# Loader
	dataloaders = {}
	dataloaders['train'] = torch.utils.data.DataLoader(
		trainset,
		batch_size=1024,
		shuffle=True,
		sampler=train_sampler,
		num_workers=0,
		)

	dataloaders['val'] = torch.utils.data.DataLoader(
		validset,
		batch_size=1024,
		shuffle=True,
		sampler=valid_sampler,
		num_workers=0,
		)

	print("Success data load...")
	print("trainset : ",len(trainset))
	print("validset : ",len(validset))

	since = time.time()
	
	val_acc_history = []

	model = Res.ResNet(Res.Bottleneck, [3,8,36,3])

	model.load_state_dict(torch.load("model/resnet152_best-2.ckpt"))

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	print('Run %s'%(device))

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	num_epochs = 200
	learning_rate = 0.0001

	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				# Set model to training mode
				model.train()
			else:
				# Set model to evaluate mode
				model.eval()
			
			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:
				optimizer.zero_grad()
				
				inputs = inputs.to(device)
				labels = labels.to(device)

				with torch.set_grad_enabled(phase == 'train'):
					if is_inception and phase == 'train':
						outputs, aux_outputs = model(inputs)
						loss1 = criterion(outputs, labels)
						loss2 = criterion(aux_outputs, labels)
						loss = loss1 + 0.4*loss2
					else:
						outputs = model(inputs)
						loss = criterion(outputs, labels)

					_, predicted = torch.max(outputs, 1)

					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(predicted == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			# Deep copy model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
				if (epoch+1) % 2 == 0:
					torch.save(model.state_dict(), 'model/resnet152_best-%s.ckpt'%(str(epoch+1)))
			if phase == 'val':
				val_acc_history.append(epoch_acc)

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	# model.load_state_dict(best_model_wts)
	# return model, val_acc_history

if __name__ == '__main__':
	train_val()
