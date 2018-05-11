from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class RRDLoss(nn.Module):
    def __init__(self, num_classes):
        super(RRDLoss, self).__init__()
        self.num_classes = num_classes

    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)      # [N,#anchors]

        num_neg = 3*pos.long().sum(1)  # [N,]
        if num_neg.sum().data[0] == 0:
            num_neg[:] = 10
        neg = rank < num_neg[:,None]   # [N,#anchors]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets, alpha=0.2):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 8].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 8].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = alpha * SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0  # [N,#anchors]
        batch_size = pos.size(0)
        num_pos = pos.sum().data[0]

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        # print(pos.shape, loc_preds.shape)
        if num_pos > 0:
            mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,8]
            loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)

        #===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        #===============================================================
        cls_loss = F.cross_entropy(cls_preds.view(-1,self.num_classes), \
                                   cls_targets.view(-1), reduce=False)  # [N*#anchors,]
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets<0] = 0  # set ignored loss to 0
        neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
        cls_loss = cls_loss[pos|neg].sum()
        num_neg = neg.sum().data[0]

        locl = loc_loss.data[0]/num_pos if num_pos > 0 else 0
        clsl = cls_loss.data[0]/num_neg
        print('loc_loss: %.3f | cls_loss: %.3f' % (locl, clsl), end=' | ')

        loss = (alpha * loc_loss + cls_loss)/(num_pos + num_neg) if num_pos > 0 else cls_loss / num_neg
        return loss
