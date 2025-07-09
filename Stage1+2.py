import datetime
import argparse
import inspect
import random
import lpips
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
from utils import save_image, EDILayer, tonemap, Gaussian_filtering, VGGLoss, wavelet_combine, wavelet_decomposition, FFTLoss, GANLoss, Sobel_pytorch, rgb_to_grayscale, local_int_align_satDark_hist_norm_RGB
from networks import OursControlNetModel, HDRev_Encoder, HDRev_Encoder_stage2
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
    # time.sleep(1.5 * 50 * 60)
    local_rank = init_dist(launcher=launcher)
    global_rank = distributed.get_rank()
    num_processes = distributed.get_world_size()
    is_main_process = (local_rank == 0)
    # create checkpoints and folders
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    folder_name = f'debug' if debug else folder_name
    out_folder = os.path.join(config.output_dir, folder_name)

    # create scheduler and models
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.noise_scheduler_kwargs))
    
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained', subfolder='vae', low_cpu_mem_usage=False, device_map=None, local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained', subfolder='tokenizer', local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained', subfolder='text_encoder', local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_path, cache_dir='/openbayes/input/input0/pretrained', subfolder='unet', low_cpu_mem_usage=False, device_map=None, local_files_only=True) # inpainting version?
    controlnet = OursControlNetModel.from_pretrained(config.pretrained_controlnet_model_path, cache_dir='/openbayes/input/input0/pretrained',
                                                 cross_attention_dim=1024 if config.noise_scheduler_kwargs.prediction_type=='v_prediction' else 768,\
                                                  conditioning_channels=config.conditioning_channels, ignore_mismatched_sizes=True, low_cpu_mem_usage=False, local_files_only=True)
    cond_encoder = HDRev_Encoder(num_bins=config.train_dataset.num_bins)
    upsampler = HDRev_Encoder_stage2(in_channels=vae.config.latent_channels)
    
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
        
        # state_dict_upsample = torch.load(pretrained, map_location='cpu')['state_dict_upsample']
        # m, u = upsampler.load_state_dict(state_dict_upsample) 
        # print(m, u)
        # print(f'upsampler:\n###### missing keys: {len(m)}; \n###### unexpected keys: {len(u)}')

    # process trainable and frozen params
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    cond_encoder.requires_grad_(False)
    upsampler.requires_grad_(False)

    if config.train_controlnet:
        for name, param in controlnet.named_parameters():
            for module_name in config.controlnet_trainable_modules:
                if module_name in name:
                    param.requires_grad = True
        for name, param in cond_encoder.named_parameters():
            param.requires_grad = True

    trainable_params = []
    upsampler.requires_grad_(True)
    trainable_params += list(filter(lambda p: p.requires_grad, upsampler.parameters()))

    # crate optimizer
    trainable_params += list(filter(lambda p: p.requires_grad, controlnet.parameters())) \
                       + list(filter(lambda p: p.requires_grad, cond_encoder.parameters()))

    # gan_loss = GANLoss().to(local_rank)
    # gan_loss.requires_grad_(True)
    # trainable_params += list(filter(lambda p: p.requires_grad, gan_loss.parameters()))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon
    )
    
    if is_main_process:
        print(f"trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # enable gradient checkpointing
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
    
    # move to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    unet.to(local_rank)
    controlnet.to(local_rank)
    cond_encoder.to(local_rank)
    upsampler.to(local_rank)

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
    
    kwargs = {'unet':unet, 'vae': vae, 'tokenizer':tokenizer, 'text_encoder':text_encoder,
              'controlnet':controlnet, 'cond_encoder': cond_encoder, 'scheduler':noise_scheduler, 'isVal':True, 
              'upsampler':upsampler}
    validation_pipeline = create_pipeline(config.validation_pipeline, kwargs).to("cuda")
    validation_pipeline.enable_vae_slicing()

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
        train_file = os.path.join('Stage1+2.py')
        os.system(f'cp {train_file} {code_dir}')
        os.system(f'cp -r networks {code_dir}')

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description('Steps')

    accept_eta = 'eta' in set(inspect.signature(noise_scheduler.step).parameters.keys())
    accept_generator = 'generator' in set(inspect.signature(noise_scheduler.step).parameters.keys())
        
    extra_step_kwargs = {}
    if accept_eta:
        extra_step_kwargs['eta'] = config.validation_setup.eta
    if accept_generator:
        extra_step_kwargs['generator'] = torch.Generator(device=local_rank)

    vgg_loss = VGGLoss().to(local_rank)
    # fft_loss = FFTLoss().to(local_rank)

    for epoch in range(first_epoch, num_train_epochs):
        dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(dataloader):
            # training
            pixel_values = tonemap(batch["gts"]).to(local_rank) * 2 - 1
            LDR = batch['pixel_images'].to(local_rank)
            EVS = batch['pixel_events'].to(local_rank)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
            
                latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            _batch_size = latents.shape[0]
            
            noise_scheduler.set_timesteps(config.noise_scheduler_kwargs.num_train_timesteps, device=local_rank)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (_batch_size,)).long()
            timesteps = noise_scheduler.timesteps[timesteps]
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps.detach().cpu()]
            # beta_prod_t = 1 - alpha_prod_t
            # weights_decay = beta_prod_t * beta_prod_t / alpha_prod_t / (1 - alpha_prod_t)

            with torch.no_grad():
                prompt_ids = tokenizer(
                    [''] * _batch_size,
                    max_length=tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
            
            condition_images, condition_list = cond_encoder(batch['pixel_images'].to(local_rank), batch['pixel_events'].to(local_rank))
            down_block_res_samples, mid_block_res_samples = controlnet(noisy_latents, timesteps, 
                                                            encoder_hidden_states=encoder_hidden_states, 
                                                            controlnet_cond=condition_list,
                                                            return_dict=False)
            model_pred = unet(noisy_latents, timesteps.to(local_rank),
                              encoder_hidden_states=encoder_hidden_states,
                              down_block_additional_residuals=down_block_res_samples,
                            #   down_intrablock_additional_residuals=down_block_res_samples,
                              mid_block_additional_residual=mid_block_res_samples
                              ).sample
            
            if noise_scheduler.config.prediction_type == 'epsilon':
                target = noise#noisy_latents - latents
            elif noise_scheduler.config.prediction_type == 'v_prediction':
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f'Unknown prediction type {noise_scheduler.config.prediction_type}')

            noise_scheduler.scale_model_input(noisy_latents, timesteps)
            original_pred_list = []
            for (model_pred_i, noisy_latents_i), timestep in zip(zip(model_pred.split(1), noisy_latents.split(1)), timesteps):
                original_pred = noise_scheduler.step(model_pred_i.detach(), timestep.detach(), noisy_latents_i.detach()).pred_original_sample.to(local_rank)
                original_pred_list.append(original_pred)
            original_pred = torch.cat(original_pred_list)
            img_pred = vae.decode(original_pred / vae.config.scaling_factor).sample

            # sub_out = torch.clamp((img_pred + 1) / 2, 0, 1)
            # sub_gt = torch.clamp((pixel_values.detach() + 1) / 2, 0, 1)
            # new_gt = local_int_align_satDark_hist_norm_RGB(sub_gt, sub_out)
            # new_gt = new_gt.to(local_rank) * 2 - 1
            
            out_img = upsampler(condition_list, original_pred, img_pred)

            noise_loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
            upsample_loss = (vgg_loss(out_img.float(), pixel_values.float()).mean() * 0.1 + \
                          F.mse_loss(out_img.float(), pixel_values.float(), reduction='mean')) * 0.1
            loss = noise_loss + upsample_loss

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            # print(optimizer.param_groups[0]['lr'])

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            # logging
            if is_main_process and (not debug) and use_wandb:
                wandb.log({'noise_loss': noise_loss.item(), 'upsampler_loss': upsample_loss.item()}, step=global_step)

            # saving
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(dataloader) - 1):
                save_path = os.path.join(out_folder, 'checkpoints')
                state_dict = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'state_dict_controlnet': controlnet.state_dict(),
                    # 'state_dict_unet': unet.module.state_dict()
                    # 'state_dict_vae': vae.state_dict(),
                    'state_dict_cond': cond_encoder.state_dict(),
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

                with torch.no_grad():
                    results, sample, latent = validation_pipeline(prompt=encoder_hidden_states, 
                                                condition_images=batch['pixel_images'], 
                                                condition_events=batch['pixel_events'], 
                                                height=height, width=width, 
                                                generators=generator, 
                                                # latents=latents,
                                                **config.validation_setup)

                # import cv2
                # from utils import tensor2im
                # cv2.imwrite('a.jpg', tensor2im(batch['pixel_images']))
                # cv2.imwrite('b.jpg', tensor2im(batch['gts']))
                # cv2.imwrite('c.jpg', tensor2im(sample))
                sub_out = torch.clamp((sample.detach() + 1) / 2, 0, 1)
                sub_gt = torch.clamp((pixel_values.detach() + 1) / 2, 0, 1)
                new_gt = local_int_align_satDark_hist_norm_RGB(sub_gt, sub_out)
                new_gt = new_gt.to(local_rank) * 2 - 1
            
                visuals = {}
                # high_lat, low_lat = wavelet_decomposition(sample)
                # high_gt, low_gt = wavelet_decomposition(pixel_values)
                # high_cnn, low_cnn = wavelet_decomposition(out_img)
                # combine = wavelet_combine(high_cnn, low_lat)
                # combine_gt = wavelet_combine(high_gt, low_lat)
                # visuals[f'{global_step}_results_combine'] = (combine + 1) / 2
                # visuals[f'{global_step}_combine_gt'] = (combine_gt + 1) / 2
                visuals[f'{global_step}_new_gt'] = (new_gt + 1) / 2
                visuals[f'{global_step}_results_tm'] = (results + 1) / 2
                visuals[f'{global_step}_results_sample'] = (sample + 1) / 2
                # visuals[f'{global_step}_results_low'] = (low_lat + 1) / 2
                visuals[f'{global_step}_events'] = batch['pixel_events']
                visuals[f'{global_step}_images'] = batch['pixel_images']
                visuals[f'{global_step}_gt'] = (pixel_values + 1) / 2
                save_path = os.path.join(out_folder, 'images')
                save_image(visuals, save_path)
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