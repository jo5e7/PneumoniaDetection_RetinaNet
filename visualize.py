import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def get_filename(path):
	print(path)
	#print(len(path))
	name = ""
	for l in range(len(path)-5, 0, -1):
		if path[l] == "/":
			break
		else:
			name = path[l] + name
			#print(name)
		pass
	return name
	pass

def visualize(csv_val, csv_classes, model):


	dataset = "csv"


	if dataset == 'csv':
		dataset_val = CSVDataset(train_file=csv_val, class_list=csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	retinanet = torch.load(model)

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	kaggle_ouput = []
	for idx, data in enumerate(dataloader_val):
		print(idx)
		kaggle_row = []
		with torch.no_grad():
			st = time.time()
			#print("data shape:", data['img'].shape)
			scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			#print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores>0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			print('Scores', scores)
			#print("name", data['name'])


			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			kaggle_row.append(get_filename(data['name'][0]))
			row = ''
			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(x1, y1, x2, y2)
				if (j==0):
					row = row + str(round(scores[j].item(),2)) +" " +str(x1) + ' ' + str(y1) + ' ' + str(x2 - x1) + ' ' + str(y2 - y1)
					pass
				else:
					row = row + " " + str(round(scores[j].item(),2)) +" "+ str(x1) + ' ' + str(y1) + ' ' + str(x2 - x1) + ' ' + str(y2 - y1)


			for ann in data['annot']:
				for annotation in ann:
					cv2.rectangle(img, (annotation[0], annotation[1]), (annotation[2], annotation[3]), color=(0, 255, 0), thickness=2)
				pass

			#cv2.imshow('img', img)
			kaggle_row.append(row)
			print(kaggle_row)
			print(idxs)
			kaggle_ouput.append(kaggle_row)
			cv2.waitKey(0)

	import pandas as pd
	pd.DataFrame(kaggle_ouput, columns=['patientId', 'PredictionString']).to_csv("/home/jdmaestre/PycharmProjects/test_kaggle.csv")


csv_classes = "/home/jdmaestre/PycharmProjects/Pneumonia_dataset/class_map.csv"
model =  "/home/jdmaestre/PycharmProjects/final_models/20ep_50res_2bs_original_lessAugmentation/model_final.pt"
model =  "/home/jdmaestre/Final models Pneumonia/NOag_FULLds/model_final.pt"
csv_val = "/home/jdmaestre/PycharmProjects/Pneumonia_dataset_synthetic/synthetic_test_set.csv"
csv_val = "/home/jdmaestre/PycharmProjects/test_labels.csv"


visualize(csv_val, csv_classes, model)