import torch
from .box import change_box_order
import shapely
from shapely.geometry import Polygon
import numpy as np


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

    rec = torch.stack([xmin, ymin, xmax, ymax], 1)
    if order == 'xywh':
        rec = change_box_order(rec, 'xyxy2xywh')
    return rec


def rec2quad(rec, order='xyxy'):
    if order == 'xywh':
        rec = change_box_order(rec, 'xywh2xyxy')

    return rec[:,[0,1,2,1,2,3,0,3]]


def _signle_iou(poly1, poly2):
    if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
        iou = 0.0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0.0
    return iou


def quad_iou(quads1, quads2):
    quads2_pol = [Polygon(item.reshape(4, 2)).convex_hull for item in quads2.numpy()]
    def multi2one(item):
        poly1 = Polygon(item.reshape(4, 2)).convex_hull
        return [_signle_iou(poly1, poly2) for poly2 in quads2_pol]
    ious = np.apply_along_axis(multi2one, axis=1, arr=quads1.numpy())
    return torch.tensor(ious)


def quad_nms(quads, scores, threshold):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,8].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    _, order = scores.sort(0, descending=True)
    polys = [Polygon(item.reshape(4, 2)).convex_hull for item in quads.numpy()]
    
    keep = []
    while order.numel() > 0: #Returns the total number of elements in the input tensor.
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        ovr = torch.FloatTensor([_signle_iou(polys[i], polys[j]) for j in order[1:]])
        ids = (ovr<=threshold).nonzero().squeeze()

        if ids.numel() == 0:
            break
        order = order[ids+1] # add 1 because of i = order[0] already use order[0] and j begin at 1

    return torch.LongTensor(keep)


def quad_clamp(quad, xmin, ymin, xmax, ymax):
    '''Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (x1,y1,x2,y2,x3,y3,x4,y4), sized [N,8].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    '''
    quad[:,0].clamp_(min=xmin, max=xmax)
    quad[:,1].clamp_(min=ymin, max=ymax)
    quad[:,2].clamp_(min=xmin, max=xmax)
    quad[:,3].clamp_(min=ymin, max=ymax)
    quad[:,4].clamp_(min=xmin, max=xmax)
    quad[:,5].clamp_(min=ymin, max=ymax)
    quad[:,6].clamp_(min=xmin, max=xmax)
    quad[:,7].clamp_(min=ymin, max=ymax)
    return quad
