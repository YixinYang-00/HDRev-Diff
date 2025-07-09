'''
this file containing pre-process code for loading dataset

To add a custom dataset A, add a file called 'A_dataset.py' and define a subclass ADataset inherited from BaseDataset.
ADataset have four functions:
    -- __init__                     initializa the class, first call BaseDataset.__init__(self, opt).
    -- __len__                      return the size of dataset
    -- __getitem__                  get a data from dataloader
    -- modify_commandline_options   optional

please specify the dataset class in command by '--dataset A'.
'''
import importlib
import torch.utils.data as data
from data_processing.base_dataset import BaseDataset

def create_dataset(opt):
    '''
        create dataset by given options
    '''
    dataset_class = find_dataset_using_name(opt.dataset_type)
    dataset = dataset_class(opt)
    print(f"dataset [{type(dataset).__name__}] was created successfully")
    print(f'The number of images in {opt.dataset_type} dataset = {len(dataset)}.')
    return dataset


def find_dataset_using_name(dataset_name):
    dataset_filename = 'data_processing.' + dataset_name + '_dataset'
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None    
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset
