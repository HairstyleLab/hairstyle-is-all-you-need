import cv2
import numpy as np

import numpy as np
from PIL import Image
from skimage import color
import os


def get_avg_a_b(lab_image):
    height, width = lab_image.shape[:2]

    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    avg_a = np.mean(a_channel)
    avg_b = np.mean(b_channel)

    return avg_a, avg_b


def shift_a_b(lab_image, a_shift, b_shift):
    shifted_lab = lab_image.copy()

    l_channel = shifted_lab[:, :, 0]
    a_channel = shifted_lab[:, :, 1]
    b_channel = shifted_lab[:, :, 2]

    luminance_scale = (l_channel / 100.0) * 1.1

    a_delta = a_shift * luminance_scale
    b_delta = b_shift * luminance_scale

    shifted_lab[:, :, 1] = a_channel + a_delta
    shifted_lab[:, :, 2] = b_channel + b_delta

    return shifted_lab


def grayworld_white_balance(image_array):
    # Handle RGBA images by converting to RGB
    if image_array.shape[2] == 4:
        print("RGBA 이미지 감지 - RGB로 변환 중...")
        # Remove alpha channel
        image_array = image_array[:, :, :3]

    rgb_normalized = image_array.astype(np.float32) / 255.0

    print("RGB -> Lab 색공간 변환 중...")
    lab_image = color.rgb2lab(rgb_normalized)

    print("평균 색상 계산 중...")
    avg_a, avg_b = get_avg_a_b(lab_image)

    print("Gray World Assumption 적용 중...")
    shifted_lab = shift_a_b(lab_image, -avg_a, -avg_b)

    print("Lab -> RGB 색공간 변환 중...")
    rgb_balanced = color.lab2rgb(shifted_lab)

    balanced_array = np.clip(rgb_balanced * 255.0, 0, 255).astype(np.uint8)

    return balanced_array