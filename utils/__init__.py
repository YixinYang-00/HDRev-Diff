import os
import cv2
import torch
import random
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
from .HDRreader import *
from .loss import *
import torch.nn.functional as F
from .histm import *

eps = 1e-6

def HDRReader(path):
    if path.endswith('.hdr'):
        hdr = cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH)[:, :, ::-1].copy()
    elif path.endswith('.exr'):
        hdr = readEXR(path)
    else:
        hdr = cv2.imread(path)[:, :, ::-1] / 255.0
    return hdr

def add_PG_noise(img, sigma_s=None, sigma_c=None):
    min_log = np.log([0.00001])
    if sigma_s is None:
        sigma_s = min_log + np.random.rand(1) * (np.log([0.01]) - min_log)
        sigma_s = np.exp(sigma_s)
    if sigma_c is None:
        sigma_c = min_log + np.random.rand(1) * (np.log([0.01]) - min_log)
        sigma_c = np.exp(sigma_c)
    # add noise
    sigma_total = np.sqrt(sigma_s * img + sigma_c)
    noisy_img =  img + \
            np.random.normal(0.0, 1.0, img.shape) * (sigma_s * img) + \
            np.random.normal(0.0, 1.0, img.shape) * sigma_c
    noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img, sigma_s, sigma_c

def func(x, a, b):
    return a * pow(x, b)

def fitRF(func_idx):
    startwith = 6 * (func_idx - 1)
    with open('dorfCurves.txt') as f:
        i = 0
        for line in f.readlines()[startwith:startwith+6]:
            if i == 3:
                I = np.fromstring(line, dtype=np.float32, sep=' ')
            if i == 5:
                B = np.fromstring(line, dtype=np.float32, sep=' ')
            i = i + 1
    
    popt, pcov = curve_fit(func, I, B, bounds=(1e-6, 10))
    a, b = popt[0], popt[1]
    return a, b, I, B

def ldr_generator(hdr, under_over_ratio, exposure=None, a=None, b=None, sig_s=None, sig_c=None, augmentation=False, fixed=False):
    if exposure is None:
        if fixed:
            seed = 1
        else:
            seed = np.random.sample()
            
        eps_gen = 0.1 / 255.0
        if seed > under_over_ratio: # over-exposure
            ratio = 1 - 0.3 * (seed - under_over_ratio) / (1 - under_over_ratio) - 0.2
            exposure = 1 / (np.percentile(hdr, ratio * 100) + eps_gen)
        else:
            ratio = 0.3 * (under_over_ratio - seed) / under_over_ratio + 0.2
            exposure = 1 / (np.percentile(hdr, ratio * 100) + eps_gen) / 255.0        # CRF
        RF_idx = np.random.randint(1, 100)
        a, b, I, B = fitRF(RF_idx)
        sig_s, sig_c = None, None

    t_img = hdr * exposure
    if augmentation:
        t_img, sig_s, sig_c = add_PG_noise(t_img, sig_s, sig_c)
    
    t_img = func(t_img, a, b)
    f_img = np.clip(t_img, 0, 1)
    i_img = (f_img * 255).astype(np.uint8)
    return (i_img / 255.0).astype(np.float32), {'exposure': exposure, 
                                                'a': a, 'b': b, 'sig_s': sig_s, 'sig_c': sig_c}

def tensor2im(input_img):
    if not isinstance(input_img, np.ndarray):
        if isinstance(input_img, torch.Tensor):
            image_tensor = input_img.data
        else:
            return input_img
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255
    else:
        image_numpy = input_img
    return image_numpy[:, :, :3]

def tonemap(img, log_sum_prev=None, return_prev=False):
    # return torch.log(1 + 500 * img) / math.log(1 + 500)
    key_fac, epsilon, tm_gamma = 0.5, 1e-6, 1.4
    XYZ = BGR2XYZ(img)
    b, c, h, w = XYZ.shape
    if log_sum_prev is None:
        log_sum_prev = torch.log(epsilon + XYZ[:, 0, :,:]).sum((1, 2), keepdim=True)
        log_avg_cur = torch.exp(log_sum_prev / (h * w))
        key = key_fac
    else:
        log_sum_cur = torch.log(XYZ[:, 1, :,:] + epsilon).sum((1, 2), keepdim=True)
        log_avg_cur = torch.exp(log_sum_cur / (h * w))
        log_avg_temp = torch.exp((log_sum_cur + log_sum_prev) / (2.0 * h * w))
        key = key_fac * log_avg_cur / log_avg_temp
        log_sum_prev = log_sum_cur
    Y = XYZ[:, 1, :, :]
    Y = Y / log_avg_cur * key
    Y = torch.log(Y + 1) # newly add
    Lmax =  torch.max(torch.max(Y, 1, keepdim=True)[0], 2, keepdim=True)[0]
    L_white2 = Lmax * Lmax
    L = Y * (1 + Y / L_white2) / (1 + Y)
    XYZ *= (L / (XYZ[:, 1, :, :] + epsilon)).unsqueeze(1)
    image = XYZ2BGR(XYZ)
    image = torch.clamp(image, 0, 1) ** (1 / tm_gamma)
    if return_prev:
        return image, log_sum_prev
    else:
        return image
        
def save_image(visuals, path, name=''):
    for label, img in visuals.items():
        print(os.path.join(path, label + '.jpg'), label, torch.mean(img), torch.min(img), torch.max(img))
        img = tensor2im(img)
        cv2.imwrite(os.path.join(path, label + '.jpg'), img[:, :, ::-1])
        if ('results_w' in label) or ('gt' in label): 
            cv2.imwrite(os.path.join(path, label + '.hdr'), img[:, :, ::-1])

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
    
def Sobel_pytorch(img):
    b, c, h, w = img.shape
    kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img.device)
    kernel = kernel.expand(c, c, -1, -1)
    Gx = F.conv2d(img, kernel, stride = 1, padding = 1) 
    Gy = F.conv2d(img, kernel.transpose(2, 3), stride = 1, padding = 1)
    return torch.stack([Gx, Gy], dim=1)

def whiteBalance(img):
    h, w = img.shape[:2]
    img = np.transpose(img, (2,0,1))
    img = np.reshape(img, (3, -1))

    r_mean = img[0].mean() + eps
    g_mean = img[1].mean() + eps
    b_mean = img[2].mean() + eps

    gray = (r_mean + g_mean + b_mean) / 3
    
    mat = [[gray/r_mean, 0, 0], [0, gray/g_mean, 0], [0,0, gray/b_mean]]
    img_wb = np.dot(mat, img)
    img_wb = np.reshape(img_wb, (3, h, w))
    img_wb = np.transpose(img_wb, (1,2,0))
    
    return img_wb

def rgb_to_grayscale(image, num_output_channels=1):
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]
    
    Y = (0.2126 * r) + (0.7152 * g) + (0.0722 * b) + eps
    Y = torch.stack([Y] * num_output_channels, dim=-3)
    return Y
    
def rgb_to_grayscale_numpy(image, num_output_channels=1):
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]
    
    Y = (0.2126 * r) + (0.7152 * g) + (0.0722 * b) + eps
    Y = np.stack([Y] * num_output_channels, axis=-3)
    return Y
    

def BGR2XYZ(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))
    
    b = image[..., 0, :, :]
    g = image[..., 1, :, :]
    r = image[..., 2, :, :]
    
    X = (0.4124 * r) + (0.3576 * g) + (0.1805 * b)
    Y = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    Z = (0.0193 * r) + (0.1192 * g) + (0.9505 * b)

    return torch.stack((X, Y, Z), -3)


def XYZ2BGR(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))
    
    X = image[..., 0, :, :]
    Y = image[..., 1, :, :]
    Z = image[..., 2, :, :]

    r = (3.240625 * X) + (-1.537208 * Y) + (-0.498629 * Z)
    g = (-0.968931 * X) + (1.875756 * Y) + (0.041518 * Z)
    b = (0.055710 * X) + (-0.204021 * Y) + (1.056996 * Z)
    
    return torch.stack((b, g, r), -3)

def paired_random_crop(img_gts, img_lqs, lq_patch_size, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        lq_patch_size (int): LQ patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = lq_patch_size

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = top, left
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def gaussian_map(h, w, mu=0, sigma=1):
    x = torch.linspace(-1, 1, h)
    y = torch.linspace(-1, 1, w)
    x, y = torch.meshgrid(x, y)
    gauss = 1 / (2 * math.pi * sigma ** 2) * torch.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma ** 2))
    return gauss.unsqueeze(0).unsqueeze(0)