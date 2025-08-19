import datetime
import argparse
import wandb
import math
import os

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.distributed as distributed

from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler

from data_processing import create_dataset
from utils import save_image, VGGLoss, GANLoss
from networks import HDRev_Encoder, OursAutoencoderKL

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

def main(name, config, use_wandb=False, debug=False, pretrained=("", "")):
    pretrained_unet, pretrained_upsample = pretrained
    device = 'cuda'
    # create checkpoints and folders
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    folder_name = f'debug' if debug else folder_name
    out_folder = os.path.join(config.output_dir, folder_name)

    # create scheduler and models
    cond_encoder = HDRev_Encoder(num_bins=config.train_dataset.num_bins)
    upsampler = OursAutoencoderKL.from_pretrained(config.pretrained_model_path, cache_dir='pretrained', subfolder='vae', low_cpu_mem_usage=False, device_map=None)
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, cache_dir='pretrained', subfolder='vae', low_cpu_mem_usage=False, device_map=None)
    
    for name, param in upsampler.named_parameters():
        if ('fusion' in name):
            # param.requires_grad = True
            if 'encode_enc_3.conv_out' in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.constant_(param,1e-6)

    if pretrained_unet != "":
        if not os.path.exists(pretrained_unet):
            raise ValueError(f'pretrained file {pretrained_unet} not exists.')
        print(f'load state dict from {pretrained_unet}')

        state_dict_cond = torch.load(pretrained_unet, map_location='cpu')['state_dict_cond']
        m, u = cond_encoder.load_state_dict(state_dict_cond) 
        print(m, u)
        print(f'cond_encoder:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')
    else :
        print('Not specify base model')
        exit()

    if pretrained_upsample != "":
        if not os.path.exists(pretrained_upsample):
            raise ValueError(f'pretrained file {pretrained_upsample} not exists.')
        print(f'load state dict from {pretrained_upsample}')

        state_dict_upsample = torch.load(pretrained_upsample, map_location='cpu')['state_dict']
        m, u = upsampler.load_state_dict(state_dict_upsample)
        print(m, u)
        print(f'upsampler:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')

    # process trainable and frozen params
    cond_encoder.requires_grad_(False)
    upsampler.requires_grad_(False)
    for name, param in upsampler.named_parameters():
        if ('fusion' in name):
            param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, upsampler.parameters()))
    # crate optimizer
    gan_loss = GANLoss().to(device)
    gan_loss.requires_grad_(True)
    trainable_params += list(filter(lambda p: p.requires_grad, gan_loss.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon
    )
    
    print(f"trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # move to GPU
    cond_encoder.to(device)
    upsampler.to(device)
    vae.to(device)

    # create dataset and dataloader
    dataset = create_dataset(config.train_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    max_train_steps = config.max_train_steps
    checkpointing_steps = config.checkpointing_steps
    gradient_accumulation_steps = config.gradient_accumulation_steps
    # diffusion iterations and learning rates
    
    lr_scheduler = get_scheduler(
        config.lr_sheduler_type,
        optimizer=optimizer, 
        num_warmup_steps=config.lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps
    )
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = config.batch_size * gradient_accumulation_steps
    
    print("***** Running training *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {config.batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")

    
    if (not debug) and use_wandb:
        wandb.init(project="HDR-diffu", name=folder_name, config=dict(config))
    
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'checkpoints'), exist_ok=True)
    OmegaConf.save(config, os.path.join(out_folder, 'config.yaml'))
    # save the code of training and the network 
    code_dir = os.path.join(out_folder, 'code')
    os.makedirs(os.path.join(out_folder, 'code'), exist_ok=True)
    train_file = os.path.join('train_upsampler.py')
    os.system(f'cp {train_file} {code_dir}')
    os.system(f'cp -r networks {code_dir}')

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description('Steps')

        
    vgg_loss = VGGLoss().to(device)

    for epoch in range(first_epoch, num_train_epochs):

        for step, batch in enumerate(dataloader):
            # training
            new_gt = batch["gts"].to(device) # 
            latents = batch['latents'].to(device)

            condition_images, condition_list = cond_encoder(batch['pixel_images'].to(device), batch['pixel_events'].to(device), return_img=False)

            out_img = upsampler.decode(latents / upsampler.config.scaling_factor, condition_list).sample

            upsample_loss = (vgg_loss(out_img, new_gt).mean() * 0.0001) + F.mse_loss(out_img.float(), new_gt.float(), reduction='mean') * 0.01#)

            optimizer.zero_grad()
            upsample_loss.backward()

            optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            # logging
            if (not debug) and use_wandb:
                wandb.log({'upsampler_loss': upsample_loss.item(), 'gan_loss': 0}, step=global_step)

            # saving
            if (global_step % checkpointing_steps == 0 or step == len(dataloader) - 1):
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
            if (global_step % config.validation_steps == 0):
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(config.global_seed)

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
            logs = {'step_loss': upsample_loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step > max_train_steps:
                break
    distributed.destroy_process_group()

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--pretrained_unet",  type=str, default="")
    parser.add_argument("--pretrained_upsample",  type=str, default="")
    parser.add_argument("--wandb",    action="store_true")
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--name",  type=str, default="")
    args = parser.parse_args()
    
    name   = Path(args.config).stem + '_' + args.name
    
    config = OmegaConf.load(args.config)

    main(name=name, use_wandb=args.wandb, config=config, debug=args.debug, pretrained=(args.pretrained_unet, args.pretrained_upsample))