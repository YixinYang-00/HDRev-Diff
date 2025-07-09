import os
import cv2
import h5py
import torch
import random
import fnmatch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from .base_dataset import BaseDataset
from utils import HDRReader, ldr_generator, tensor2im, EventSlicer, Sobel_pytorch, whiteBalance, paired_random_crop, EDILayer
 
eps = 1e-8
 
class evsImageH5SingleDataset(BaseDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
 
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
 
        Returns:
            the modified parser.
        """
        if is_train:
            parser.add_argument('--train_size', type=int, default=16, help='the size n of training windows, [n * n]')
            
        return parser
 
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.num_bins = opt.num_bins
        self.event_representation = opt.event_representation
        self.event_norm = opt.event_norm
        if self.isTrain:
            self.patch_size = opt.patch_size
            self.under_over_ratio = opt.under_over_ratio
            
        self.files, self.video_names = [], []
        if not self.isTrain:
            self.File = h5py.File(opt.dataroot, 'r')['/mnt/Nas-CP-share/sherry/datasets/desc/test/']
        else:
            self.File = h5py.File(opt.dataroot, 'r')['/openbayes/input/input0/train_collect/']
        self.files = list(self.File.keys())
        self.MAX_LENGTHS = opt.max_dataset_size
 
        self.files = self.files[:self.MAX_LENGTHS]
 
        self.height, self.width = opt.patch_size
        self.dataset_size = len(self.files)
 
    def __len__(self):        
        return self.dataset_size   
     
    def __getitem__(self, index):
        idx = index % self.dataset_size
        video_name = self.files[idx]
        File = self.File[video_name]
        img = np.array(File['new_gt'])

        ldr = np.array(File['ldr'])

        event_representation = np.array(File['evs'])
        latents = np.array(File['latents'])
 
        event_representation = torch.from_numpy(event_representation)
        ldr = torch.from_numpy(ldr).float()
        img = torch.from_numpy(img).float()
        latents = torch.from_numpy(latents).float()#[0]
        return {
            'save_path': '-'.join([video_name, f'{idx}']),
            'pixel_events': event_representation,
            'pixel_images': ldr,
            'latents': latents,
            'gts': img,
            'text': "", 
            'negative_text': "blur, haze, deformed iris, deformed pupilssss, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation, over-exposure, under-exposure, heavily noise"#''
        }
 
    def __events_to_voxel_grid(self, x, y, p, t, num_bins, width, height):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
 
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        """
 
        assert(num_bins > 0)
        assert(width > 0)
        assert(height > 0)
 
        voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
        if len(x) == 0:
            return voxel_grid
 
        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = t[-1]
        first_stamp = t[0]
        deltaT = last_stamp - first_stamp
 
        if deltaT == 0:
            deltaT = 1.0
 
        mask = (x < width) & (x >= 0) & (y < height) & (y >= 0)
 
        t = (num_bins - 1) * (t - first_stamp) / deltaT
        ts = t[mask]
        xs = x.astype(np.int)[mask]
        ys = y.astype(np.int)[mask]
        pols = p[mask]
        pols[pols == 0] = -1  # polarity should be +1 / -1
 
        tis = ts.astype(np.int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts
 
        valid_indices = tis < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
                + tis[valid_indices] * width * height, vals_left[valid_indices])
 
        valid_indices = (tis + 1) < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
                + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])
 
        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
 
        return voxel_grid