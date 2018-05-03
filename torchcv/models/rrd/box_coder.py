'''Encode object boxes and labels.'''
import math
import torch
import itertools

from torchcv.utils import meshgrid
from torchcv.utils.box import box_iou, box_nms, change_box_order
from torchcv.utils.quadrilateral import bounding, rec2quad, quad_nms


class RRDBoxCoder:
    def __init__(self, rrd_model):
        self.steps = rrd_model.steps
        self.box_sizes = rrd_model.box_sizes
        self.aspect_ratios = rrd_model.aspect_ratios
        self.fm_sizes = rrd_model.fm_sizes
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * self.steps[i]
                cy = (h + 0.5) * self.steps[i]

                s = self.box_sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(self.box_sizes[i] * self.box_sizes[i+1])
                boxes.append((cx, cy, s, s))

                s = self.box_sizes[i]
                for ar in self.aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.

        RRD coding rules:
          for i = 1, 2, 3, 4:
          tx_i = (x_i - anchor_x_i) / (anchor_w)
          ty_i = (y_i - anchor_y_i) / (anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (x1,y1,x2,y2,x3,y3,x4,y4), sized [#obj, 8].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,8].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1][0]
            return (i[j], j)

        default_boxes = self.default_boxes  # xywh
        default_boxes = change_box_order(default_boxes, 'xywh2xyxy')

        ious = box_iou(default_boxes, bounding(boxes))  # [#anchors, #obj]
        index = torch.LongTensor(len(default_boxes)).fill_(-1)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0
            masked_ious[:,j] = 0

        mask = (index<0) & (ious.max(1)[0]>=0.5)
        if mask.any():
            index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        default_boxes = rec2quad(default_boxes)

        loc_targets = (boxes-default_boxes) / (self.default_boxes[:,2:].repeat(1,4))
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index<0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh1=0.5, nms_thresh2=0.2):
        default_boxes = rec2quad(self.default_boxes, 'xywh')
        box_preds = self.default_boxes[:,2:].repeat(1,4) * loc_preds + default_boxes

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes-1):
            score = cls_preds[:,i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]
            # stage 1
            keep1 = box_nms(bounding(box), score, nms_thresh1)
            box = box[keep1]
            score = score[keep1]
            # stage 2
            keep2 = quad_nms(box, score, nms_thresh2)
            boxes.append(box[keep2])
            labels.append(torch.LongTensor(len(keep2)).fill_(i))
            scores.append(score[keep2])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores
