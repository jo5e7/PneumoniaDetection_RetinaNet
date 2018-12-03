from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from future.utils import raise_from

from PIL import Image


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        name = self.image_names[idx]
        #print(name)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot, 'name': name}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    names = [s['name'] for s in data]
    #print(names)
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'name': names}


def collater_image_only(data):
    imgs = [data]

    for s in imgs:
        print("s", s[0][0].shape)

    widths = [int(s[0][0].shape[0]) for s in imgs]
    heights = [int(s[0][0].shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i][0][0]

        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img




    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    print("s", padded_imgs.shape)
    return padded_imgs


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'name':sample['name']}

class Resizer_only_img(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image = sample

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        print("new image shape:", new_image.shape)


        return torch.from_numpy(new_image)

class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'name': sample['name']}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots, 'name':sample['name']}

class Normalizer_only_image(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        sample = np.array(sample)



        #sample = skimage.io.imread(sample)

        if len(sample.shape) == 2:
            sample = skimage.color.gray2rgb(sample)

        sample = sample.astype(np.float32)/255.0

        image = ((sample.astype(np.float32)-self.mean)/self.std)
        print("shape:", image.shape)
        return image

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

import numbers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import functional as F
from skimage.viewer import ImageViewer
from skimage.transform import rotate as skRotate
from copy import deepcopy

class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """


        #v = ImageViewer(new_sample['img'])

        #v.show()

        #print("-------------")
        if random.random() < 0.3:
            #print("rotation", random.random())
            angle = self.get_params(self.degrees)
            new_sample = deepcopy(sample)
            # print(sample['img'].shape)

            # print("axis:", x_axis, y_axis)

            # print(len(sample['annot']))
            # print(sample['annot'])
            # img = sample['img']
            # fig, ax = plt.subplots(1)
            # ax.imshow(img)
            for n in range(0, len(new_sample['annot'])):
                annot = new_sample['annot'][n]
                # print(annot[0])
                x_axis = (((annot[2] - annot[0])) / 2) + annot[0]
                y_axis = (((annot[3] - annot[1])) / 2) + annot[1]
                old_x1 = annot[0] - x_axis
                old_y1 = annot[1] - y_axis
                old_x2 = annot[2] - x_axis
                old_y2 = annot[3] - y_axis
                old_x3 = annot[0] - x_axis
                old_y3 = annot[3] - y_axis
                old_x4 = annot[2] - x_axis
                old_y4 = annot[1] - y_axis
                # rect = patches.Rectangle((annot[0], annot[1]), annot[2]-annot[0], annot[3]-annot[1], linewidth=1, edgecolor='g', facecolor='none')
                # ax.add_patch(rect)

                new_x1 = (old_x1 * np.cos(np.deg2rad(angle))) - (old_y1 * np.sin(np.deg2rad(angle)))
                new_y1 = (old_y1 * np.cos(np.deg2rad(angle))) + (old_x1 * np.sin(np.deg2rad(angle)))
                new_x2 = (old_x2 * np.cos(np.deg2rad(angle))) - (old_y2 * np.sin(np.deg2rad(angle)))
                new_y2 = (old_y2 * np.cos(np.deg2rad(angle))) + (old_x2 * np.sin(np.deg2rad(angle)))
                new_x3 = (old_x3 * np.cos(np.deg2rad(angle))) - (old_y3 * np.sin(np.deg2rad(angle)))
                new_y3 = (old_y3 * np.cos(np.deg2rad(angle))) + (old_x3 * np.sin(np.deg2rad(angle)))
                new_x4 = (old_x4 * np.cos(np.deg2rad(angle))) - (old_y4 * np.sin(np.deg2rad(angle)))
                new_y4 = (old_y4 * np.cos(np.deg2rad(angle))) + (old_x4 * np.sin(np.deg2rad(angle)))
                new_x1 += x_axis
                new_x2 += x_axis
                new_y1 += y_axis
                new_y2 += y_axis
                new_x3 += x_axis
                new_x4 += x_axis
                new_y3 += y_axis
                new_y4 += y_axis

                # print("angle:", angle, "x1:", old_x1+512, "y1: ", old_y1+512, "x1':", new_x1, "y1':", new_y1)
                xs = [new_x1, new_x2, new_x3, new_x4]
                ys = [new_y1, new_y2, new_y3, new_y4]

                # print(xs)
                # print(min(xs))
                # print(ys)
                # print(max(ys))
                # new_x1 = min(xs)
                # new_y1 = min(ys)
                # new_x2 = max(xs)
                # new_y2 = max(ys)
                # rect = patches.Rectangle((new_x1, new_y1), new_x2-new_x1, new_y2-new_y1, linewidth=1, edgecolor='r', facecolor='none')
                # ax.add_patch(rect)

                annot = [min(xs), min(ys), max(xs), max(ys), annot[4]]
                new_sample['annot'][n] = annot
                # print(annot)
                # print("-------------")
                pass
            # plt.show()

            new_sample['img'] = skRotate(new_sample['img'], angle)
            return new_sample
        return sample



    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """



        #im = new_sample['img']
        #ax = plt.subplot(2, 1, 1)
        #ax.imshow(im)
        #for n in range(0, len(new_sample['annot'])):
        #    annot = new_sample['annot'][n]
        #    rect = patches.Rectangle((annot[0], annot[1]), annot[2]-annot[0], annot[3]-annot[1], linewidth=1, edgecolor='g', facecolor='none')
        #    ax.add_patch(rect)
        #    pass
        #ax = plt.subplot(2, 1, 2)

        #ax.imshow(new_sample['img'])



        #plt.show()
        if random.random() < self.p:
            new_sample = deepcopy(sample)
            x_axis = new_sample['img'].shape[0] / 2
            new_sample['img'] = np.fliplr(new_sample['img'])
            for n in range(0, len(new_sample['annot'])):
                annot = new_sample['annot'][n]
                annot[0] = x_axis - (annot[0] - x_axis)
                annot[2] = x_axis - (annot[2] - x_axis)

                # rect = patches.Rectangle((annot[0], annot[1]), annot[2] - annot[0], annot[3] - annot[1], linewidth=1,
                #                         edgecolor='r', facecolor='none')
                # ax.add_patch(rect)
                new_sample['annot'][n] = annot
                pass

            return new_sample
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

from scipy import ndimage
class Blur(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        if random.random() < self.p:
            new_sample = deepcopy(sample)
            new_sample['img'] = ndimage.uniform_filter(new_sample['img'], size=(4, 4, .4))
            #show_images(sample, new_sample)
            return new_sample
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

from skimage.exposure import adjust_gamma
import skimage.io as io
class Gamma_Correction(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            new_sample = deepcopy(sample)
            new_sample['img'] = adjust_gamma(new_sample['img'], gamma=0.8, gain=0.9)
            #show_images(sample, new_sample)
            return new_sample
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


from skimage.util import random_noise
class Image_Noise(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """



        #ax = plt.subplot(1,2,1)
        #ax.imshow(new_sample['img'])
        #ax = plt.subplot(1, 2, 2)
        #ax.imshow(sample['img'])
        #plt.show()

        if random.random() < self.p:
            new_sample = deepcopy(sample)
            new_sample['img'] = random_noise(new_sample['img'], var=0.0005)
            #show_images(sample, new_sample)
            return new_sample
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def show_images(original, new):
    ax = plt.subplot(1,2,1)
    for n in range(0, len(new['annot'])):
       annot = new['annot'][n]
       rect = patches.Rectangle((annot[0], annot[1]), annot[2]-annot[0], annot[3]-annot[1], linewidth=1, edgecolor='g', facecolor='none')
       ax.add_patch(rect)
       pass
    ax.imshow(new['img'])
    ax.set_title("Modified image")
    ax = plt.subplot(1, 2, 2)
    for n in range(0, len(original['annot'])):
       annot = original['annot'][n]
       rect = patches.Rectangle((annot[0], annot[1]), annot[2]-annot[0], annot[3]-annot[1], linewidth=1, edgecolor='r', facecolor='none')
       ax.add_patch(rect)
       pass
    ax.imshow(original['img'])
    ax.set_title("Original image")
    plt.show()