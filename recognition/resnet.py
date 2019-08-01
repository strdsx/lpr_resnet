import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import os
import cv2

def conv3x3(in_planes, out_planes, stride=1):
	### 3x3 convolution with padding ###
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
	### 1x1 convolution ###
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
	expansion=1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

# euc-kr palte character classes = 52
class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes=84, zero_init_residual=False):
		super(ResNet, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

# Updata Learning rate
def update_lr(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr



def main():
	# train datasets transforms
	# transforms.Pad(4)
	transforms_train = transforms.Compose([
		transforms.Resize((32,32)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
		])

	# train loader
	trainset = torchvision.datasets.ImageFolder(
		root='data/rename_gray_headline_piap',
		transform=transforms_train)
	
	train_loader = torch.utils.data.DataLoader(
		trainset,
		batch_size=1024,
		shuffle=True,
		num_workers=4)

	# ResNet-50 : [3, 4, 6, 3]
	# ResNet-101 : [3, 4, 23, 3]
	# ResNet-152 : [3, 8, 36, 3]
	model = ResNet(Bottleneck, [3, 8, 36, 3])
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print("Run %s"%device)
	if torch.cuda.device_count() > 1:
		print('\n===> Training on GPU!')
		model = nn.DataParallel(model)
	num_epochs = 200
	learning_rate = 0.0001

	# Loss & Optimizer
	criterion = nn.CrossEntropyLoss()
	# criterion = nn.BCEWithLogitsLoss()
	criterion.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

	total_step = len(train_loader)
	curr_lr = learning_rate

	for i, (images, labels) in enumerate(train_loader, 0):
		images=images.to(device)
		labels=labels.to(device)

	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader, 0):
			optimizer.zero_grad()

			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)

			loss = criterion(outputs, labels)

			# Backward and optimize
			loss.backward()
			optimizer.step()

			if (i+1) % 15 == 0:
				print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

		# Decay learning rate
		if (epoch+1) % 20 == 0:
			curr_lr /= 3
			update_lr(optimizer, curr_lr)

		if (epoch+1) % 2 == 0:
			num = str(epoch + 1)
			torch.save(model.state_dict(), 'model/resnet152-'+num+'.ckpt')

	torch.save(model.state_dict(), 'model/resnet152.ckpt')

def test():
	net = ResNet(Bottleneck, [3, 8, 36, 3])
	y = net(torch.randn(1,3,32,32))
	print(y.size())


if __name__ == '__main__':
	main()
