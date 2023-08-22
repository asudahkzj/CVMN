import numpy as np
from PIL import Image
import cv2


def resize(image, limit_size, interpolation=None):
    image = Image.fromarray(image)
    if image.width < image.height:
        scale = float(limit_size) / float(image.height)
        new_width = int(image.width * scale + 1)
        if new_width > limit_size:
            new_width = limit_size
        image = np.array(cv2.resize(np.array(image),
                                    (new_width, limit_size), interpolation=interpolation)).astype(np.float32)
    else:
        scale = float(limit_size) / float(image.width)
        new_height = int(image.height * scale + 1)
        if new_height > limit_size:
            new_height = limit_size
        image = np.array(cv2.resize(np.array(image),
                                    (limit_size, new_height), interpolation=interpolation)).astype(np.float32)
    return image


def resize_and_pad(image, limit_size, interpolation=None):
    image = resize(image, limit_size, interpolation=interpolation)
    mask = np.ones([limit_size, limit_size], dtype=np.uint8)
    # print(image.shape)
    if image.shape[0] == limit_size:
        left_pad = (limit_size - image.shape[1]) >> 1
        right_pad = limit_size - image.shape[1] - left_pad
        pad_config = [[0, 0], [left_pad, right_pad]]
        if len(image.shape) == 3:
            pad_config.append([0, 0])
        image = np.pad(image, pad_config, mode='constant', constant_values=0)
        mask[:, 0:left_pad] = 0
        mask[:, limit_size - right_pad:] = 0
    else:
        left_pad = (limit_size - image.shape[0]) >> 1
        right_pad = limit_size - image.shape[0] - left_pad
        pad_config = [[left_pad, right_pad], [0, 0]]
        if len(image.shape) == 3:
            pad_config.append([0, 0])
        image = np.pad(image, pad_config, mode='constant', constant_values=0)
        mask[0:left_pad, :] = 0
        mask[limit_size - right_pad:, :] = 0
    return image, mask, left_pad
