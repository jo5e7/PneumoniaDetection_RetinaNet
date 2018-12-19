import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import Blur, Gamma_Correction, RandomHorizontalFlip, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, RandomRotation, Normalizer, Image_Noise
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def train(csv_train=None, csv_classes=None, csv_val=None, epochs=12, depth=50, batch_size=2):

	dataset = "csv"

	# Create the data loaders
	if dataset == 'csv':

		if csv_train is None:
			raise ValueError('Must provide --csv_train when training on COCO,')

		if csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on COCO,')


		dataset_train = CSVDataset(train_file=csv_train, class_list=csv_classes, transform=transforms.Compose([RandomRotation(6),Gamma_Correction(0.45), Image_Noise(0.45), Blur(0.45) , Normalizer(), Augmenter(), Resizer()]))

		if csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=csv_val, class_list=csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

	# Create the model
	if depth == 18:
		retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	elif depth == 34:
		retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	elif depth == 50:
		retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	elif depth == 101:
		retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	elif depth == 152:
		retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
	
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	retinanet.module.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	# Change
	total_loss_data = []
	class_loss_data = []
	reg_loss_data = []
	# Change

	for epoch_num in range(epochs):

		retinanet.train()
		retinanet.module.freeze_bn()


		epoch_loss = []

		# Change
		epoch_reg_loss = []
		epoch_class_loss = []
		# Change


		for iter_num, data in enumerate(dataloader_train):
			try:
				optimizer.zero_grad()

				classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()

				loss = classification_loss + regression_loss
				
				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				# Change
				epoch_reg_loss.append(float(regression_loss))
				epoch_class_loss.append(float(classification_loss))
				# Change

				print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
				
				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue


		if dataset == 'csv' and csv_val is not None:

			print('Evaluating dataset')

			mAP = csv_eval.evaluate(dataset_val, retinanet)

		# Change
		total_loss_data.append(np.mean(epoch_loss))
		class_loss_data.append(np.mean(epoch_class_loss))
		reg_loss_data.append(np.mean(epoch_reg_loss))
		print("Epoch loss", total_loss_data)
		print("Epoch loss - classification", class_loss_data)
		print("Epoch loss - Regression", reg_loss_data)
		with open("Output.txt", "w") as text_file:
			print("------------------------", file=text_file)
			print("Epoch loss", total_loss_data, file=text_file)
			print("Epoch loss - classification", class_loss_data, file=text_file)
			print("Epoch loss - Regression", reg_loss_data, file=text_file)
		# Change
		scheduler.step(np.mean(epoch_loss))	

		torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(dataset, epoch_num))

	retinanet.eval()

	torch.save(retinanet, 'model_final.pt'.format(epoch_num))



	# Change
	import matplotlib.pyplot as plt
	plt.plot(total_loss_data, label='Total loss')
	plt.plot(class_loss_data, label='Classification loss')
	plt.plot(reg_loss_data, label='Regression loss')
	plt.ylabel("Loss")
	plt.xlabel("Epoch")
	plt.title("Epoch losses")
	plt.legend()
	plt.show()
	#Change


from pathlib import Path
mypath = Path().absolute()
print(mypath)

csv_train = str(mypath) + "/google_cloud/stage_2_train_labels.csv"
csv_train = str(mypath) + "/google_cloud/synthetic_train_set.csv"
csv_classes = str(mypath) + "/google_cloud/class_map.csv"
epochs = 100
depth = 50

train(csv_train, csv_classes, epochs=epochs, depth=depth, batch_size=8)