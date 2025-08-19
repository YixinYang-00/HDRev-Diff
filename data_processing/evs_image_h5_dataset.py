import os
import h5py
import torch
import numpy as np
import torch.nn.functional as tF
from .base_dataset import BaseDataset
from utils import ldr_generator, whiteBalance, paired_random_crop

eps = 1e-8

class evsImageH5Dataset(BaseDataset):
    
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
            
        self.files, self.video_names, self.video_file = [], [], []
        videos = [os.path.join(opt.dataroot, name) for name in os.listdir(os.path.join(opt.dataroot)) if name.endswith('.h5')]
        F = h5py.File(videos[0], 'r')
        samples = list(F.keys())
        F.close()
        for video_name in videos:
            num_samples = len(samples) - opt.video_length
            self.files += [samples[i:i+opt.video_length] for i in range(1, num_samples)]
            F = h5py.File(video_name, 'r')
            self.video_file += [F for i in range(1, num_samples)]
            self.video_names += [video_name for i in range(1, num_samples)]
        self.MAX_LENGTHS = opt.max_dataset_size

        self.files = self.files[:self.MAX_LENGTHS]

        self.dataset_size = len(self.files)
        if self.isTrain:
            self.height, self.width =  (self.patch_size, self.patch_size) if isinstance(self.patch_size, int) \
                                      else self.patch_size
        else:
            self.height, self.width = opt.patch_size

    def __len__(self):        
        return self.dataset_size   
    
    def __getitem__(self, index):
        idx = index % self.dataset_size
        video_name = self.video_names[idx]
        f_index = self.files[idx]
        F = self.video_file[idx]
        
        events, gts, ldrs = [], [], []
        ldr_params = {'exposure': None}
        for file_idx in f_index[:1]:
            
            img = np.array(F[file_idx]['gt'])
            img = whiteBalance(img)
            img = np.transpose(img, (2, 0, 1))
            img = img / np.mean(img) * 0.1

            if self.isTrain:
                ldr, ldr_params = ldr_generator(img, self.under_over_ratio, augmentation=True, **ldr_params)
            else:
                ldr = np.array(F[file_idx]['ldr'])
                ldr = np.transpose(ldr, (2, 0, 1))
            event_representation = np.array(F[file_idx]['evs_to_prev1'])
            if self.event_norm:
                # Normalize the event tensor (voxel grid) so that
                # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
                mean, stddev = event_representation[event_representation != 0].mean(), event_representation[event_representation != 0].std()
                event_representation[event_representation != 0] = (event_representation[event_representation != 0] - mean) / stddev
            
            event_representation = torch.from_numpy(event_representation)
            
            ldr = torch.from_numpy(ldr).float()
            img = torch.from_numpy(img).float()
            
            if self.isTrain:
                img, [event_representation, ldr] = paired_random_crop(img.permute(1, 2, 0), [event_representation.permute(1, 2, 0), ldr.permute(1, 2, 0)], self.height, None)
                img, event_representation, ldr = img.permute(2, 0, 1), event_representation.permute(2, 0, 1), ldr.permute(2, 0, 1)
            else:
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