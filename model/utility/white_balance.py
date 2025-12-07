import cv2
import numpy as np


def gray_world(image):
    # image: uint8 (H,W,3)
    image = image.astype(np.float32)

    mean_global = image.mean()
    mean_channel = image.mean(axis=(0, 1))

    scale = mean_global / mean_channel  # (3,)

    result = image * scale
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def gamma_correction(img, gamma=1.0):
    if gamma == 1.0:
        return img
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv) * 255
    return cv2.LUT(img, table.astype(np.uint8))

def white_balance(img):
    img = gray_world(img)
    img = gamma_correction(img, gamma=3)
    return img