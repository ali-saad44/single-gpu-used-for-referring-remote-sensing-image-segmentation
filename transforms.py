import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, target):
        image = F.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, (self.h, self.w), interpolation=Image.NEAREST)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)  # Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1)
        image = F.resize(image, size)
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class TargetAwareZoomCrop(object):
    """
    Zoom-In Crop Strategy for small targets.
    If the target is extremely small (e.g., < 10% of image), it dynamically crops
    a high-resolution region around the target during training. This trains the model
    to handle extremely small objects without increasing global image size/VRAM.
    """
    def __init__(self, zoom_prob=0.5, small_threshold=0.1):
        self.zoom_prob = zoom_prob
        self.small_threshold = small_threshold

    def __call__(self, image, target):
        if random.random() > self.zoom_prob:
            return image, target
            
        # Target is a PIL Image
        target_np = np.array(target)
        pos = np.where(target_np > 0)
        if len(pos[0]) == 0:
            return image, target
            
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        
        target_area = (ymax - ymin) * (xmax - xmin)
        img_area = target_np.shape[0] * target_np.shape[1]
        
        if target_area / max(1, img_area) < self.small_threshold:
            # Crop tightly around the object with some padding
            pad_y = int((ymax - ymin) * random.uniform(1.0, 3.0))
            pad_x = int((xmax - xmin) * random.uniform(1.0, 3.0))
            
            center_y = (ymin + ymax) // 2
            center_x = (xmin + xmax) // 2
            
            crop_h = ymax - ymin + 2 * pad_y
            crop_w = xmax - xmin + 2 * pad_x
            max_side = max(crop_h, crop_w, 128) # Ensure reasonable crop size
            
            crop_ymin = max(0, center_y - max_side // 2)
            crop_ymax = min(target_np.shape[0], center_y + max_side // 2)
            crop_xmin = max(0, center_x - max_side // 2)
            crop_xmax = min(target_np.shape[1], center_x + max_side // 2)
            
            image = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
            target = target.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
            
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
        return image, target


class RandomAffine(object):
    def __init__(self, angle, translate, scale, shear, resample=0, fillcolor=None):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, image, target):
        affine_params = T.RandomAffine.get_params(self.angle, self.translate, self.scale, self.shear, image.size)
        image = F.affine(image, *affine_params)
        target = F.affine(target, *affine_params)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

