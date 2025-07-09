import datetime
import argparse
import inspect
import random
import lpips
import h5py
import wandb
import math
import time
import os

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.distributed as distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import DDIMScheduler, LMSDiscreteScheduler, AutoencoderKL
from diffusers.models import UNet2DConditionModel, ControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.models.lora import LoRAConv2dLayer, LoRALinearLayer

from data_processing import create_dataset
from utils import save_image, EDILayer, tonemap, Gaussian_filtering, VGGLoss, wavelet_combine, wavelet_decomposition, local_int_align_satDark_hist_norm_RGB
from networks import OursControlNetModel, HDRev_Encoder, HDRev_Encoder_stage2, OursAutoencoderKL
# from networks.controlnetCross import regiter_attention_editor_diffusers
# from networks.attention_processor import EVSFeatProj
from pipeline import create_pipeline

eps = 1e-8

def get_condition_input(config, batch):
    if config.control_type == 'evs':
        ret = batch['pixel_events']
    elif config.control_type == 'ldr':
        ret = batch['pixel_images']
    elif config.control_type == 'evs+ldr':
        ret = torch.cat([batch['pixel_events'], batch['pixel_images']], dim=1)
    else:
        raise NotImplementedError(f'Not implemented control type')
    return ret

def main(exp_name, config, pretrained, debug, save):
    device = 'cuda'
    time.sleep(0.8 * 60 * 60)
    # create scheduler and models
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.noise_scheduler_kwargs))
    
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained/', subfolder='vae', low_cpu_mem_usage=False, device_map=None, local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained/', subfolder='tokenizer', local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained/', subfolder='text_encoder', local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained/', subfolder='unet', low_cpu_mem_usage=False, device_map=None, local_files_only=True) # inpainting version?
    controlnet = OursControlNetModel.from_pretrained(config.pretrained_controlnet_model_path, cache_dir='/openbayes/input/input0/pretrained/',
                                                 cross_attention_dim=1024 if config.noise_scheduler_kwargs.prediction_type=='v_prediction' else 768,\
                                                  conditioning_channels=config.conditioning_channels, ignore_mismatched_sizes=True, low_cpu_mem_usage=False, local_files_only=True)
    # regiter_attention_editor_diffusers(controlnet)
    cond_encoder = HDRev_Encoder(num_bins=config.test_dataset.num_bins)
    upsampler_wstruct = OursAutoencoderKL.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained', subfolder='vae', low_cpu_mem_usage=False, device_map=None, local_files_only=True)
    upsampler_wostruct = OursAutoencoderKL.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained', subfolder='vae', low_cpu_mem_usage=False, device_map=None, local_files_only=True)
    # evs_feat_proj = EVSFeatProj(config.conditioning_channels, 768)
    
    if pretrained != "":
        if not os.path.exists(pretrained):
            raise ValueError(f'pretrained file {pretrained} not exists.')
        print(f'load state dict from {pretrained}')

        state_dict_controlnet = torch.load(pretrained, map_location='cpu')['state_dict_controlnet']
        m, u = controlnet.load_state_dict(state_dict_controlnet) 
        print(m, u)
        print(f'controlnet:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')
    
        state_dict_cond = torch.load(pretrained, map_location='cpu')['state_dict_cond']
        m, u = cond_encoder.load_state_dict(state_dict_cond) 
        print(m, u)
        print(f'cond_encoder:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')
        
        state_dict_upsample = torch.load('checkpoints/Train_upsampler_continueVgg0-2025-05-15T12-35-58/checkpoints/latest.ckpt', map_location='cpu')['state_dict_upsample']
        # state_dict_upsample = torch.load(pretrained, map_location='cpu')['state_dict_upsample']
        m, u = upsampler_wostruct.load_state_dict(state_dict_upsample) 
        print(m, u)
        print(f'upsampler:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')

        state_dict_upsample = torch.load('checkpoints/Train_upsampler_continueVgg00001-2025-05-15T17-00-16/checkpoints/latest.ckpt', map_location='cpu')['state_dict_upsample']
        # state_dict_upsample = torch.load(pretrained, map_location='cpu')['state_dict_upsample']
        m, u = upsampler_wstruct.load_state_dict(state_dict_upsample) 
        print(m, u)
        print(f'upsampler:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')

        # state_dict_evsfeatproj = torch.load(pretrained, map_location='cpu')['state_dict_evsfeatproj']
        # m, u = evs_feat_proj.load_state_dict(state_dict_evsfeatproj) 
        # print(m, u)
        # print(f'evs_feat_proj:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')
    # move to GPU
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    controlnet.to(device)
    cond_encoder.to(device)
    upsampler_wostruct.to(device)
    upsampler_wstruct.to(device)
    # evs_feat_proj.to(device)

    # create dataset and dataloader
    dataset = create_dataset(config.test_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True
    )

    kwargs = {'unet':unet, 'vae': vae, 'tokenizer':tokenizer, 'text_encoder':text_encoder,
              'controlnet':controlnet, 'cond_encoder': cond_encoder, 'scheduler':noise_scheduler, 'isVal':False, 
              'upsampler_wo':upsampler_wostruct, 'upsampler_w':upsampler_wstruct, 'upsampler':None}
    pipeline = create_pipeline(config.validation_pipeline, kwargs).to(device)
    pipeline.enable_vae_slicing()
    
    # create checkpoints and folders
    folder_name = exp_name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    folder_name = f'debug' if debug else folder_name
    out_folder = os.path.join(config.output_dir, folder_name)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'images'), exist_ok=True)
    OmegaConf.save(config, os.path.join(out_folder, 'config.yaml'))
    if save:
        save_F = h5py.File(os.path.join(out_folder, 'inputs_w_latents.h5'), 'w')

    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description('Steps')

    accept_eta = 'eta' in set(inspect.signature(noise_scheduler.step).parameters.keys())
    accept_generator = 'generator' in set(inspect.signature(noise_scheduler.step).parameters.keys())
        
    extra_step_kwargs = {}
    if accept_eta:
        extra_step_kwargs['eta'] = config.validation_setup.eta
    if accept_generator:
        extra_step_kwargs['generator'] = torch.Generator(device=device)

    for step, batch in tqdm(enumerate(dataloader)):
        # test
        generator = torch.Generator(device=device)
        generator.manual_seed(config.global_seed * step)

        height = batch['pixel_images'].shape[-2]
        width = batch['pixel_images'].shape[-1]
        
        with torch.no_grad():
                prompt_ids = tokenizer(
                    [''],
                    max_length=tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).input_ids.to(device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]


        # condition_images, condition_list, evs_feat = cond_encoder(batch['pixel_images'].to(device), batch['pixel_events'].to(device))
        # hidden_states = evs_feat_proj(evs_feat)
        for i in range(1):
            sample, results, latents = pipeline(prompt=encoder_hidden_states, 
                            condition_images=batch['pixel_images'], 
                            condition_events=batch['pixel_events'], 
                            height=height, width=width, 
                            generators=generator, 
                            **config.validation_setup)
            # continue
            if pipeline.upsampler_w is not None:
                results_w, results_wo = results
            if save:
                if debug:
                    exit()
                G = save_F.create_group(batch['save_path'][0])
                G.create_dataset('ldr', data = batch['pixel_images'][0].cpu(), compression = "gzip")
                G.create_dataset('evs', data = batch['pixel_events'][0].cpu(), compression = "gzip")
                G.create_dataset('latents', data = latents[0].cpu(), compression = "gzip")
                # G.create_dataset('sample', data = sample[0].cpu(), compression = "gzip")

                pixel_values = tonemap(batch['gts']).to(device) * 2 - 1
                G.create_dataset('gt', data = pixel_values[0].cpu(), compression = "gzip")
                sub_out = torch.clamp((sample.detach() + 1) / 2, 0, 1)
                sub_gt = torch.clamp((pixel_values.detach() + 1) / 2, 0, 1)
                new_gt = local_int_align_satDark_hist_norm_RGB(sub_gt, sub_out)
                new_gt = new_gt.to(device) * 2 - 1
                G.create_dataset('new_gt', data = new_gt[0].cpu(), compression = "gzip")
            else:
                visuals = {}
                pixel_values = tonemap(batch['gts']).to(device) * 2 - 1
                sub_out = torch.clamp((sample.detach() + 1) / 2, 0, 1)
                sub_gt = torch.clamp((pixel_values.detach() + 1) / 2, 0, 1)
                new_gt = local_int_align_satDark_hist_norm_RGB(sub_gt, sub_out)
                new_gt = new_gt.to(device) * 2 - 1

                # visuals[f'{step}_{i}_results_tm'] = (results + 1) / 2
                visuals[f'{step}_{i}_results_wo'] = (results_wo + 1) / 2
                visuals[f'{step}_{i}_results_w'] = ((results_w + 1) / 2) 
                visuals[f'{step}_{i}_results_sample'] = (sample + 1) / 2
                # high_lat, low_lat = wavelet_decomposition(sample)
                # high_cnn, low_cnn = wavelet_decomposition(results)
                # combine = wavelet_combine(high_cnn, low_lat)
                # visuals[f'{step}_{i}_results_combine'] = (combine + 1) / 2
                visuals[f'{step}_{i}_events'] = batch['pixel_events']
                visuals[f'{step}_{i}_images'] = batch['pixel_images']
                visuals[f'{step}_{i}_lin_gt'] = batch['gts']#(pixel_values + 1) / 2
                visuals[f'{step}_{i}_new_gt'] = (new_gt + 1) / 2
                save_path = os.path.join(out_folder, 'images')
                save_image(visuals, save_path, batch['save_path'][0].split('/')[-1])
            # exit()
    if save:
        save_F.close()

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--pretrained",  type=str, default="")
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--save",  action="store_true")
    parser.add_argument("--name",  type=str, default="")
    args = parser.parse_args()
    
    name   = Path(args.config).stem + '_' + args.name
    config = OmegaConf.load(args.config)

    main(exp_name=name, config=config, pretrained=args.pretrained, debug=args.debug, save=args.save)
