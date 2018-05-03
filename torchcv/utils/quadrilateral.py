import torch
from .box import change_box_order


def bounding(quad_boxes, order='xyxy'):
    '''
    Args:
      quad_boxes: (tensor) bounding boxes of (x1,y1,x2,y2,x3,y3,x4,y4), sized [N,8].
      order: (str) either 'xyxy' or 'xywh'.

    Returns:
      (tensor) get bounding rectangle boxes, sized [N,4].
    '''

    x = quad_boxes[:, [0,2,4,6]]
    y = quad_boxes[:, [1,3,5,7]]

    xmin, _ = x.min(1)
    xmax, _ = x.max(1)
    ymin, _ = y.min(1)
    ymax, _ = y.max(1)

    rec = torch.cat([xmin, ymin, xmax, ymax], 1)
    if order == 'xywh':
        rec = change_box_order(rec, 'xyxy2xywh')
    return rec


def rec2quad(rec, order='xyxy'):
    if order == 'xywh':
        rec = change_box_order(rec, 'xyhw2xyxy')

    return rec[:,[0,1,2,1,2,3,0,3]]