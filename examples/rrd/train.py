from __future__ import print_function

import sys
sys.path.append('../../')
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable

from torchcv.models.rrd import RRD, RRDBoxCoder

from torchcv.loss import RRDLoss
from torchcv.datasets.textdataset import TextDataset
from torchcv.transforms import resize_quad, random_distort, random_paste_quad, random_crop_quad, random_flip_quad


img_size = 384
batch_size = 32

train_label_files = '/Users/filick/projects/text/data/ICPR_text_train_part2_20180313/txt_9000'
train_image_files = '/Users/filick/projects/text/data/ICPR_text_train_part2_20180313/image_9000'
test_label_files = '/Users/filick/projects/text/data/train_1000/image_1000'
test_image_files = '/Users/filick/projects/text/data/train_1000/txt_1000'

checkpoints = 'checkpoint/ckpt.pth'
resume = False
INPUT_WORKERS = 16


# Model
print('==> Building model..')
net = RRD(num_classes=2)
#net = FPNSSD512(num_classes=2)
#net.load_state_dict(torch.load(args.model))
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(checkpoints)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# Dataset
print('==> Preparing dataset..')
box_coder = RRDBoxCoder(net)
def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste_quad(img, boxes, max_ratio=4, fill=(123,116,103))
    img, boxes, labels = random_crop_quad(img, boxes, labels)
    img, boxes = resize_quad(img, boxes, size=(img_size,img_size), random_interpolation=True)
    img, boxes = random_flip_quad(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

def transform_test(img, boxes, labels):
    img, boxes = resize_quad(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

trainset = TextDataset(img_root = train_image_files,
                       label_root = train_label_files,
                       transform=transform_train)

testset = TextDataset(img_root = test_image_files,
                      label_root = test_image_files,
                      transform=transform_test)

trainloader =  data.DataLoader(trainset, batch_size=batch_size,
                               shuffle=True, num_workers=INPUT_WORKERS)
testloader =  data.DataLoader(testset, batch_size=batch_size,
                              shuffle=False, num_workers=INPUT_WORKERS)

net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

# opt
lr =1e-4
momentum=0.9
weight_decay=1e-4

criterion = RRDLoss(num_classes=2)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets, names) in enumerate(trainloader):
        #print(batch_idx/len(trainloader))
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)

        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f [%d/%d]'
              % (loss.data[0], test_loss/(batch_idx+1), batch_idx+1, len(testloader)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.dirname(checkpoint)):
            os.mkdir(os.path.dirname(checkpoint))
        torch.save(state, checkpoint)
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)