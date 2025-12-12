# Modified from https://github.com/megvii-research/TLC/blob/main/basicsr/models/archs/restormer_arch.py
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F
from model.SAFMN.basicsr.utils.download_util import load_file_from_url
from model.SAFMN.basicsr.utils.colorfix import wavelet_reconstruction
from model.SAFMN.basicsr.archs.safmn_arch import SAFMN

def get_high_resolution(img, model=None):
    """
    Super-resolution using SAFMN model

    Args:
        img: Input image tensor
        model: Pre-loaded SAFMN model (if None, will load on-the-fly)

    Returns:
        High-resolution image as numpy array
    """
    # If model is not provided, load it (backward compatibility)
    if model is None:
        model_path = 'https://github.com/sunny2109/SAFMN/releases/download/v0.1.0/SAFMN_L_Real_LSDIR_x2.pth'
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=2)

        model_path = load_file_from_url(url=model_path, model_dir=os.path.join('pretrained_models'), progress=True, file_name=None)
        model.load_state_dict(torch.load(model_path)['params'], strict=True)

        model.eval()
        model = model.to(device)

    # Get device from model
    device = next(model.parameters()).device
    img = img.to(device)

    with torch.no_grad():
        output = model(img)

    img = F.interpolate(img, scale_factor=2, mode='bilinear')
    output = wavelet_reconstruction(output, img)

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))
    # if output.ndim == 3:
    #     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    return output