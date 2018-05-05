import torch
import random

from PIL import Image


def random_flip_quad(img, quads):
    '''Randomly flip PIL image.

    If boxes is not None, flip boxes accordingly.

    Args:
      img: (PIL.Image) image to be flipped.
      quads: (tensor) object boxes, sized [#obj,8].

    Returns:
      img: (PIL.Image) randomly flipped image.
      boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        if quads is not None:
            quads[:, [0,2,4,6]] *= -1
            quads[:, [0,2,4,6]] += w
            quads = quads[:, [2,3,0,1,6,7,4,5]]

    return img, quads
