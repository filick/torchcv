from __future__ import print_function

import os
import sys
import random
import math

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from PIL import Image


def _counterwise_step(v):
    return [v[i] for i in range(2,8)] + [v[0], v[1]]


def _euclidean_dist(v, xmin, xmax, ymin, ymax):
    dist = 0
    dist += math.sqrt((v[0] - xmin)**2 + (v[1] - ymin)**2)
    dist += math.sqrt((v[2] - xmax)**2 + (v[3] - ymin)**2)
    dist += math.sqrt((v[4] - xmax)**2 + (v[5] - ymax)**2)
    dist += math.sqrt((v[6] - xmin)**2 + (v[7] - ymax)**2)
    return dist


def _sort_vertices(vertices):
    xmin = min(vertices[0], vertices[2], vertices[4], vertices[6])
    xmax = max(vertices[0], vertices[2], vertices[4], vertices[6])
    ymin = min(vertices[1], vertices[3], vertices[5], vertices[7])
    ymax = max(vertices[1], vertices[3], vertices[5], vertices[7])

    min_dist = _euclidean_dist(vertices, xmin, xmax, ymin, ymax)
    best_order = vertices

    for i in range(3):
        vertices = _counterwise_step(vertices)
        dist = _euclidean_dist(vertices, xmin, xmax, ymin, ymax)
        if dist < min_dist:
            min_dist = dist
            best_order = vertices

    return best_order



class TextDataset(data.Dataset):
    '''Load image/labels/boxes from a folder.

    image file and related label file should have the same name
    '''
    def __init__(self, img_root, label_root, transform=None, return_text=False):

        self.img_root = img_root
        self.transform = transform
        self.return_text = return_text

        self.fnames = [name[:-4] for name in os.listdir(label_root)]
        self.boxes = []
        self.labels = []

        for img_id, fname in enumerate(self.fnames):
            bboxes = []
            labels = []
            for line in open(os.path.join(self.img_root, fname + '.txt'), 'r', encoding='UTF-8').read().splitlines():
                items = line.split(',')
                bboxes.append(_sort_vertices([float(i) for i in items[:8]]))
                labels.append(items[-1])
            self.boxes.append(bboxes)
            self.labels.append(labels)


    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.img_root, fname + '.jpg')
        
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.return_text:
            valid_gen = filter(lambda i: labels[i] != '###', range(len(labels)))
            boxes = [boxes[i] for i in valid_gen]
            labels = [labels[i] for i in labels]

        classes = [0] * len(labels)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, classes)

        if self.return_text:
            return image, boxes, classes, labels
        else:
            return image, boxes, classes


    def __len__(self):
        return len(self.fnames)
