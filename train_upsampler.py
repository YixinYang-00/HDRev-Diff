import datetime
import argparse
import inspect
import random
import lpips
import wandb
import time
import math
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
from utils import save_image, EDILayer, tonemap, Gaussian_filtering, VGGLoss, wavelet_combine, wavelet_decomposition, FFTLoss, GANLoss
from networks import OursControlNetModel, HDRev_Encoder, HDRev_Encoder_stage2, OursAutoencoderKL
from pipeline import create_pipeline

eps = 1e-8

def init_dist(launcher='slurm', backend='nccl'):
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend=backend)
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        distributed.init_process_group(backend=backend)
    else:
        raise NotImplementedError(f'Not implemented launcher type {launcher}')

    return local_rank


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

def main(name, launcher, config, use_wandb=False, debug=False, pretrained=""):
    # time.sleep(1.5 * 60 * 60)
    local_rank = init_dist(launcher=launcher)
    global_rank = distributed.get_rank()
    num_processes = distributed.get_world_size()
    is_main_process = (local_rank == 0)
    # create checkpoints and folders
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    folder_name = f'debug' if debug else folder_name
    out_folder = os.path.join(config.output_dir, folder_name)

    # create scheduler and models
    cond_encoder = HDRev_Encoder(num_bins=config.train_dataset.num_bins)
    upsampler = OursAutoencoderKL.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained', subfolder='vae', low_cpu_mem_usage=False, device_map=None, local_files_only=True)
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained/', subfolder='vae', low_cpu_mem_usage=False, device_map=None, local_files_only=True)
    
    for name, param in upsampler.named_parameters():
        if ('fusion' in name):
            # param.requires_grad = True
            if 'encode_enc_3.conv_out' in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.constant_(param,1e-6)

    if pretrained != "":
        if not os.path.exists(pretrained):
            raise ValueError(f'pretrained file {pretrained} not exists.')
        print(f'load state dict from {pretrained}')

        state_dict_cond = torch.load(pretrained, map_location='cpu')['state_dict_cond']
        m, u = cond_encoder.load_state_dict(state_dict_cond) 
        print(m, u)
        print(f'cond_encoder:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')
        
        # state_dict_upsample = torch.load('checkpoints/Train_upsampler_vgg0-2025-05-13T06-09-31/checkpoints/latest.ckpt', map_location='cpu')['state_dict_upsample']
        # # state_dict_upsample = torch.load(pretrained, map_location='cpu')['state_dict_upsample']
        # m, u = upsampler.load_state_dict(state_dict_upsample) 
        # print(m, u)
        # print(f'upsampler:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')
    else :
        print('Not specify base model')
        exit()

    # process trainable and frozen params
    cond_encoder.requires_grad_(False)
    upsampler.requires_grad_(False)
    # cond_encoder.netDecoder.requires_grad_(True)
    for name, param in upsampler.named_parameters():
        if ('fusion' in name):
            param.requires_grad = True
    # upsampler.requires_grad_(True)

            # print(name, torch.max(param))

    trainable_params = list(filter(lambda p: p.requires_grad, upsampler.parameters()))
    # trainable_params += list(cond_encoder.netDecoder.parameters())
    # crate optimizer
    gan_loss = GANLoss().to(local_rank)
    gan_loss.requires_grad_(True)
    trainable_params += list(filter(lambda p: p.requires_grad, gan_loss.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon
    )
    
    if is_main_process:
        print(f"trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # move to GPU
    cond_encoder.to(local_rank)
    upsampler.to(local_rank)
    vae.to(local_rank)

    # create dataset and dataloader
    dataset = create_dataset(config.train_dataset)
    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=config.global_seed
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = config.batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=config.num_workers,
        drop_last=True
    )

    max_train_steps = config.max_train_steps
    checkpointing_steps = config.checkpointing_steps
    gradient_accumulation_steps = config.gradient_accumulation_steps
    # diffusion iterations and learning rates
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(dataloader)
    
    if checkpointing_steps == -1:
        assert checkpointing_epoch != -1
        checkpointing_steps = checkpointing_epoch * len(dataloader)
    
    lr_scheduler = get_scheduler(
        config.lr_sheduler_type,
        optimizer=optimizer, 
        num_warmup_steps=config.lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps
    )
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = config.batch_size * num_processes * gradient_accumulation_steps
    
    if is_main_process:
        print("***** Running training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {config.batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_train_steps}")

    
    if is_main_process and (not debug) and use_wandb:
        wandb.init(project="HDR-diffu", name=folder_name, config=dict(config))
    
    if is_main_process:
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(os.path.join(out_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(out_folder, 'checkpoints'), exist_ok=True)
        OmegaConf.save(config, os.path.join(out_folder, 'config.yaml'))
        # save the code of test and pipeline
        code_dir = os.path.join(out_folder, 'code')
        os.makedirs(os.path.join(out_folder, 'code'), exist_ok=True)
        train_file = os.path.join('train_upsampler.py')
        os.system(f'cp {train_file} {code_dir}')
        os.system(f'cp -r networks {code_dir}')

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description('Steps')

        
    vgg_loss = VGGLoss().to(local_rank)
    fft_loss = FFTLoss().to(local_rank)

    for epoch in range(first_epoch, num_train_epochs):
        dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            # training
            # print(torch.max(batch["gts"]), torch.min(batch["gts"]))
            new_gt = batch["gts"].to(local_rank) # 
            # new_gt = tonemap(batch["gts"]).to(local_rank) * 2 - 1
            latents = batch['latents'].to(local_rank)
            # img_pred = batch['sample'].to(local_rank)
            condition_images, condition_list = cond_encoder(batch['pixel_images'].to(local_rank), batch['pixel_events'].to(local_rank), return_img=False)
            # import cv2
            # from utils import tensor2im
            # cv2.imwrite('a.jpg', tensor2im(img))
            # exit()
            # print(latents.shape, img_pred.shape, pixel_values.shape)
            # print(img_pred.shape)
            # img = img * 2 - 1
            out_img = upsampler.decode(latents / upsampler.config.scaling_factor, condition_list).sample
            # print(torch.max(out_img), torch.min(out_img), torch.max(new_gt), torch.min(new_gt))
            loss_gan = 0 #gan_loss(out_img, new_gt, 0)
            # upsample_loss = (vgg_loss(out_img, new_gt).mean() * 0.01) + 
            # upsample_loss = F.mse_loss(out_img.float(), new_gt.float(), reduction='mean') * 0.01# \
            upsample_loss = (vgg_loss(out_img, new_gt).mean() * 0.0001) + F.mse_loss(out_img.float(), new_gt.float(), reduction='mean') * 0.01#)
            # out_img = img           
            loss = upsample_loss# + loss_gan# * 0.1# + loss_gan
            # print(global_step, torch.max(pixel_values), torch.min(pixel_values), torch.max(img_pred), torch.min(img_pred), loss)
            # print(upsample_loss, loss_gan)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            # print(upsampler.decoder.fusion_layer_3.encode_enc_3.conv1.bias[0], upsampler.decoder.fusion_layer_3.encode_enc_3.conv1.bias.grad[0])
            optimizer.step()
            # exit()
            # print(optimizer.param_groups[0])

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            # logging
            if is_main_process and (not debug) and use_wandb:
                wandb.log({'upsampler_loss': upsample_loss.item(), 'gan_loss': 0}, step=global_step)

            # saving
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(dataloader) - 1):
                save_path = os.path.join(out_folder, 'checkpoints')
                state_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'state_dict_upsample': upsampler.state_dict()
                }
                if step == len(dataloader) - 1:
                    if epoch % 10 == 0:
                        torch.save(state_dict, os.path.join(save_path, f'epoch-{epoch+1}.ckpt'))
                else:
                    torch.save(state_dict, os.path.join(save_path, 'latest.ckpt'))
                print(f'saving model to {save_path} with global step {global_step}')
            
            # validation
            if is_main_process and (global_step % config.validation_steps == 0):
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(config.global_seed)

                height = config.train_dataset.patch_size if isinstance(config.train_dataset.patch_size, int) else config.train_dataset.patch_size[0]
                width = config.train_dataset.patch_size if isinstance(config.train_dataset.patch_size, int) else config.train_dataset.patch_size[1]

                img_pred = vae.decode(latents / vae.config.scaling_factor).sample
                visuals = {}
                visuals[f'{global_step}_results_tm'] = (out_img + 1) / 2
                visuals[f'{global_step}_results_sample'] = (img_pred + 1) / 2
                visuals[f'{global_step}_results_diff'] = ((out_img - new_gt) + 2) / 4
                visuals[f'{global_step}_events'] = batch['pixel_events']
                visuals[f'{global_step}_images'] = batch['pixel_images']
                visuals[f'{global_step}_gt'] = (new_gt + 1) / 2
                save_path = os.path.join(out_folder, 'images')
                save_image(visuals, save_path)
                # exit()
            logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step > max_train_steps:
                break
    distributed.destroy_process_group()

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--name",  type=str, default="")
    args = parser.parse_args()
    
    name   = Path(args.config).stem + '_' + args.name
    
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, config=config, debug=args.debug, pretrained=args.pretrained)