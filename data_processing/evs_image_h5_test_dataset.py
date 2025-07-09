import os
import cv2
import h5py
import torch
import random
import fnmatch
import numpy as np
import torch.nn.functional as tF
from torchvision import transforms
from .base_dataset import BaseDataset
from utils import HDRReader, ldr_generator, tensor2im, EventSlicer, Sobel_pytorch, whiteBalance, paired_random_crop, EDILayer

eps = 1e-8

class evsImageH5TestDataset(BaseDataset):
    
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
        self.dataroot = opt.dataroot
        if self.isTrain:
            self.patch_size = opt.patch_size
            self.under_over_ratio = opt.under_over_ratio
            
        self.files, self.video_names = [], []
        self.File = h5py.File(opt.dataroot, 'r')
        videos = list(self.File.keys())
        for video_name in videos:
            samples = list(self.File[video_name].keys())
            num_samples = min(len(samples) - opt.video_length, 2000000)
            self.files += [samples[i:i+opt.video_length] for i in range(1, num_samples)]
            self.video_names += [video_name for i in range(1, num_samples)]
        self.MAX_LENGTHS = opt.max_dataset_size

        self.files = self.files[:self.MAX_LENGTHS]#[6047:]
        self.video_names = self.video_names[:self.MAX_LENGTHS]#[6047:]
        self.dataset_size = len(self.files)
        if self.isTrain:
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.patch_size)
            ])
            self.height, self.width =  (self.patch_size, self.patch_size) if isinstance(self.patch_size, int) \
                                      else self.patch_size
        else:
            self.height, self.width = opt.patch_size
        self.data_transforms = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):        
        return self.dataset_size   
    
    def rectify_events(self, x: np.ndarray, y: np.ndarray, video_name):
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[video_name]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def __getitem__(self, index):
        idx = index % self.dataset_size
        video_name = self.video_names[idx]
        f_index = self.files[idx]
        # print(video_name)
        F = self.File[video_name]
        
        events, gts, ldrs, prompts = [], [], [], []
        ldr_params = {'exposure': None}
        for file_idx in f_index:
            if 'gt' in F[file_idx].keys():
                img = np.array(F[file_idx]['gt'])
                img = whiteBalance(img)
                img = np.transpose(img, (2, 0, 1))
                img = img / np.mean(img) * 0.1

            if 'dsec' in self.dataroot:
                ldr = np.array(F[file_idx]['ldr']) 
            elif 'HJ' in self.dataroot:
                ldr = np.array(F[file_idx]['ldr']) / 255.0
                ldr = np.transpose(ldr, (2, 0, 1))
            else:
                ldr = np.array(F[file_idx]['ldr']) 
            img = ldr

            if self.isTrain:
                ldr, ldr_params = ldr_generator(img, self.under_over_ratio, augmentation=True, **ldr_params)
            
            if 'HJ' in self.dataroot:
                event_representation = np.array(F[file_idx]['evs'])[-100000:, :]
            else:
                event_representation = np.array(F[file_idx]['evs'])
            ev_height, ev_width = int(np.max(event_representation[:, 2])), int(np.max(event_representation[:, 1]))
            t, x, y, p = event_representation[:, 0], event_representation[:, 1], event_representation[:, 2], event_representation[:, 3]
            if self.event_representation == 'voxel_grid':
                event_representation = self.__events_to_voxel_grid(x, y, p, t, self.num_bins, ev_width, ev_height)
            if self.event_norm:
                # Normalize the event tensor (voxel grid) so that
                # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
                mean, stddev = event_representation[event_representation != 0].mean(), event_representation[event_representation != 0].std()
                event_representation[event_representation != 0] = (event_representation[event_representation != 0] - mean) / stddev
            
            event_representation = torch.from_numpy(event_representation)
            
            ldr = torch.from_numpy(ldr).float()
            img = torch.from_numpy(img).float()
            
            img = tF.interpolate(img.unsqueeze(0), (self.height, self.width))[0]
            event_representation = tF.interpolate(event_representation.unsqueeze(0), (self.height, self.width))[0]
            ldr = tF.interpolate(ldr.unsqueeze(0), (self.height, self.width))[0]

            events.append(event_representation)
            gts.append(img)
            ldrs.append(ldr)
            
        return {
            'save_path': '-'.join([video_name, f'{idx}']),
            'pixel_events': events[0],
            'pixel_images': ldrs[0],
            'gts': gts[0],
            'text': "", 
            'negative_text': ""
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
        xs = x.astype(int)[mask]
        ys = y.astype(int)[mask]
        pols = p[mask]
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.astype(int)
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
