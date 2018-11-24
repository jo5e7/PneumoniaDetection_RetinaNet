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

from dataloader import CocoDataset, CSVDataset, collater_image_only, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Resizer_only_img, Normalizer_only_image


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def predict(folder_path, model_path):

    dataset_val = datasets.ImageFolder(folder_path,  transform=transforms.Compose([Normalizer_only_image(), Resizer_only_img()]))
    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, collate_fn=collater_image_only, num_workers=1, batch_size=1)
    print(dataloader_val.dataset)

    #for asd in dataloader_val:
    #    print(asd)
    #    pass



    retinanet = torch.load(model_path)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):
        print(idx)
        print("shape data:", data[0].shape)
        with torch.no_grad():
            st = time.time()
            dataa = data[0]
            dataa = dataa.view(1,3,640,640)
            print("shape dataaa:", dataa.shape)
            scores, classification, transformed_anchors = retinanet(dataa.cuda().float())
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores>0.5)
            img = np.array(255 * unnormalize(dataa[0, :, :, :])).copy()



            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            import matplotlib.pyplot as plt
            plt.imshow(img, cmap='gray')
            #plt.show()

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                #label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), 'Opacity')

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                #print(label_name)

            cv2.imshow('img', img)
            cv2.waitKey(2000)
