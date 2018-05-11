'''This random crop strategy is described in paper:
   [1] SSD: Single Shot MultiBox Detector
'''
import math
import torch
import random

from PIL import Image
from torchcv.utils.box import box_iou
from torchcv.utils.quadrilateral import quad_clamp, bounding


def random_crop_quad(
        img, quads, labels,
        min_scale=0.3,
        max_aspect_ratio=2.):
    '''Randomly crop a PIL image.

    Args:
      img: (PIL.Image) image.
      quads: (tensor) bounding boxes, sized [#obj, 8].
      labels: (tensor) bounding box labels, sized [#obj,].
      min_scale: (float) minimal image width/height scale.
      max_aspect_ratio: (float) maximum width/height aspect ratio.

    Returns:
      img: (PIL.Image) cropped image.
      boxes: (tensor) object boxes.
      labels: (tensor) object labels.
    '''
    imw, imh = img.size
    params = [(0, 0, imw, imh)]  # crop roi (x,y,w,h) out
    quads_bound = bounding(quads)
    min_iou = torch.Tensor([0, 0.1, 0.3, 0.5, 0.7, 0.9])
    min_ciou = torch.Tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    for _ in range(200):
        remain_iou = torch.sum(min_iou < 1) > 0
        remain_ciou = torch.sum(min_ciou < 1) > 0
        if not (remain_ciou or remain_iou):
            break

        scale = random.uniform(min_scale, 1)
        aspect_ratio = random.uniform(
            max(1/max_aspect_ratio, scale*scale),
            min(max_aspect_ratio, 1/(scale*scale)))
        w = int(imw * scale * math.sqrt(aspect_ratio))
        h = int(imh * scale / math.sqrt(aspect_ratio))

        x = random.randrange(imw - w)
        y = random.randrange(imh - h)

        roi = torch.FloatTensor([[x,y,x+w,y+h]])
        ious = box_iou(quads_bound, roi)
        
        if remain_ciou:
            G = (quads_bound[:,2]-quads_bound[:,0]) * (quads_bound[:,3]-quads_bound[:,1])
            B = w * h
            S = (G + B) / (ious + 1)
            cious = S / G
            if torch.sum(min_ciou <= cious.min()) > 0:
                idx = (min_ciou <= cious.min()).nonzero()[-1]
                params.append((x,y,w,h))
                min_ciou[idx] = 1.1
                continue

        if remain_iou:
            if torch.sum(min_iou <= ious.min()) > 0:
                idx = (min_iou <= ious.min()).nonzero()[-1]
                params.append((x,y,w,h))
                min_iou[idx] = 1.1
                continue

    x,y,w,h = random.choice(params)
    img = img.crop((x,y,x+w,y+h))

    center = (quads_bound[:,:2] + quads_bound[:,2:]) / 2
    mask = (center[:,0]>x) & (center[:,0]<x+w) \
         & (center[:,1]>y) & (center[:,1]<y+h)
    if mask.any():
        idx = mask.nonzero().squeeze()
        quads = quads[idx,:] - torch.FloatTensor([x,y,x,y,x,y,x,y])
        quads = quad_clamp(quads,0,0,w,h)
        labels = labels[idx]
    else:
        quads = torch.FloatTensor([[0,0,0,0,0,0,0,0]])
        labels = torch.LongTensor([0])
    return img, quads, labels
