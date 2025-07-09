import torch
import inspect
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torchvision.transforms.functional as vF

import cv2
import math
from utils import tensor2im, Gaussian_filtering, rgb_to_grayscale
import numpy as np


class Stagepipeline(DiffusionPipeline):

    def __init__(self, vae, text_encoder, tokenizer, unet, controlnet, cond_encoder, scheduler, upsampler,
                 upsampler_wo=None, upsampler_w=None,
                 isVal=False):

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            cond_encoder=cond_encoder,
            upsampler=upsampler,
            upsampler_w=upsampler_w,
            upsampler_wo=upsampler_wo
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.isVal = isVal
    
    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def _encode_prompt(self, prompt, device, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_embeddings = self.text_encoder(text_input_ids.to(device))[0]

        if not self.isVal:
            uncond_input_ids = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            uncond_text_embedding = self.text_encoder(uncond_input_ids.to(device))[0]
        
            return torch.cat([uncond_text_embedding, text_embeddings])
        else :
            return text_embeddings

    def decode_latents(self, latents, image_latents=None):
        latents = 1 / self.vae.config.scaling_factor * latents
        img = self.vae.decode(latents).sample
        return img

    def encode_latents(self, image):
        latents = self.vae.encode(image.to(self.vae.device)).latent_dist
        latents = latents.sample()
            
        latents = latents * self.vae.config.scaling_factor
        return latents

    def prepare_mask_and_masked_image_and_latents(self, image, over_exposure_value=0.9, under_exposure_value=0.01):
        assert over_exposure_value > under_exposure_value, "The over exposure pixel value must be greater than under exposure value"
        gray_image = vF.rgb_to_grayscale(image, num_output_channels=1)
        mask = (gray_image < over_exposure_value)# & (gray_image > under_exposure_value)
        mask = mask.type(torch.FloatTensor).to(image.device)
        # mask = (0.5 - torch.maximum(torch.abs(gray_image - 0.5), torch.ones_like(gray_image) * (over_exposure_value - 0.5))) / (1 - over_exposure_value)
        # mask = 1 - mask.type(torch.FloatTensor).to(self.vae.device)
        
        masked_image = mask * image

        masked_image_latents = self.vae.encode(masked_image.to(self.vae.device)).latent_dist.sample() * self.vae.config.scaling_factor

        b, c, h, w = mask.shape
        mask_resized = torch.nn.functional.interpolate(
            mask, size=(h // self.vae_scale_factor, w // self.vae_scale_factor)
        )

        return mask, masked_image, masked_image_latents, mask_resized

    def prepare_latents(self, batch_size, num_channels, height, width, device, generators, latents=None):
        shape = (batch_size, num_channels, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generators, list) and len(generators) != batch_size:
            raise ValueError(f'the length of generators {len(generators)} is not capable of batch size {batch_size}')
        if latents is None:
            if isinstance(generators, list):
                latents = [torch.randn(shape, generator=generators[i], device=device)
                            for i in range(len(generators))]
                latents = torch.cat(latents, dim=0)
            else:
                latents = torch.randn(shape, generator=generators, device=device)
        else:
            if latents.shape != shape:
                raise ValueError(f'the shape of the init latents ({latents.shape}) is not matched with the expected latent shape {shape}')
            latents = latents.to(device)
        return latents * self.scheduler.init_noise_sigma, latents

    def prepare_extra_step_kwargs(self, generator, eta):
        accept_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        accept_generator = 'generator' in set(inspect.signature(self.scheduler.step).parameters.keys())
        
        extra_step_kwargs = {}
        if accept_eta:
            extra_step_kwargs['eta'] = eta
        if accept_generator:
            extra_step_kwargs['generator'] = generator

        return extra_step_kwargs

    def next_step(self, model_output, timestep, x):
        # inverse sampling from MasaCtrl
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(self, model_output, timestep, x):
        # denoise process from MasaCtrl
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    def invertion(self, image, text_embeddings, num_inference_steps, guidance_scale=1.0):
        batch_size = image.shape[0]
        device = self.unet.device

        # prepare latents
        latents = self.encode_latents(image * 2 - 1)
        
        timestamps = self.scheduler.timesteps
        start_latents = latents
        latents_list = [start_latents]
        for t in reversed(timestamps):
            if self.isVal:
                model_input = latents
            else:
                model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeddings).sample

            if not self.isVal:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            st_latents = latents
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            
        latents_list = latents_list[::-1]
        return latents, latents_list

    @torch.no_grad()
    def __call__(self, prompt, condition_images, condition_events, height, width, num_inference_steps, 
                negative_prompt=None, eta=0.0, generators=None, latents=None, guidance_scale=1.0, latents_noise=None,
                gt=None, use_image_prompt=False, prompt_image=None, return_latents=False):
        # height = 
        # width = 

        if guidance_scale < 1:
            raise ValueError(f'guidance scale should greater than one, while the input is {guidance_scale}')
        batch_size = prompt.shape[0]
        # if latents is not None:
        #     batch_size = latents.shape[0]
        # elif isinstance(prompt, list):
        #     batch_size = len(prompt)
        # else:
        #     batch_size = 1

        device = self.unet.device

        # encode prompt
        # prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        # if negative_prompt is not None:
        #     negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
        text_embeddings = prompt if self.isVal else torch.cat([prompt, prompt])#self._encode_prompt(prompt, device, negative_prompt)
        # if use_image_prompt:
        #     image_prompt_embeds, uncond_image_prompt_embeds = self.image_encoder.get_image_embeds(prompt_image)
        #     if self.isVal:
        #         text_embeddings = torch.cat([text_embeddings, image_prompt_embeds],dim=1) 
        #     else:
        #         text_embeds, uncond_text_embeds = text_embeddings.chunk(2)
        #         text_embeddings = torch.cat([text_embeds, image_prompt_embeds, uncond_text_embeds, uncond_image_prompt_embeds],dim=1) 

        condition_images = condition_images.to(device)
        condition_events = condition_events.to(device)

        # import time
        # a1 = time.time()
        conditions, condition_list = self.cond_encoder(condition_images, condition_events)
        # conditions, _, condition_list = self.cond_encoder(condition_images, condition_events, return_img=True)
        # condition = torch.cat([condition_images, condition_events], dim=1)
        # sum_flops, sum_parms = 0, 0
        # from thop import profile
        # flops, params = profile(self.cond_encoder, inputs=(condition_images, condition_events))
        # sum_flops += flops
        # sum_parms += params

        # prepare timestamps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timestamps = self.scheduler.timesteps

        # prepare mask
        # mask, masked_image, masked_image_latents, mask_resized = self.prepare_mask_and_masked_image_and_latents(condition_images)

        # prepare latents
        num_channels_latents = self.vae.config.latent_channels
        if latents_noise is None:
            latents_noise, noise = self.prepare_latents(batch_size, num_channels_latents, height, width, device, generators)
        # latents, latents_list = self.invertion(condition_images / torch.mean(condition_images) * 0.5, text_embeddings, num_inference_steps, guidance_scale)
        
        latents = latents_noise# * (1 - mask_resized) + latents * mask_resized
        # flops, params = profile(self.controlnet.controlnet_cond_embedding, inputs=(condition_list, ))
        # sum_flops += flops
        # sum_parms += params
        # condition = self.controlnet.controlnet_cond_embedding(condition_list)
        # ldr = condition_images.to(device)
        # from utils import BGR2YCbCr
        # Y_ldr = BGR2YCbCr(ldr)[:, 0, :, :]
        # attention_mask = (0.5 - torch.maximum(torch.abs(Y_ldr - 0.5), torch.ones_like(Y_ldr) * (0.8 - 0.5))) / (1 - 0.8)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generators, eta)
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timestamps):
                if self.isVal:
                    latent_input = latents
                    controlnet_emb = text_embeddings
                else:
                    latent_input = torch.cat([latents, latents], dim=0)
                    controlnet_emb = text_embeddings.chunk(2)[1]
                latent_model_input = self.scheduler.scale_model_input(latent_input, t)
                controlnet_input = self.scheduler.scale_model_input(latents, t)
                down_block_res_samples, mid_block_res_samples = self.controlnet(controlnet_input, t, 
                                                                           encoder_hidden_states=controlnet_emb, 
                                                                           controlnet_cond=condition_list,
                                                                           return_dict=False)
                # flops, params = profile(self.controlnet, inputs=(controlnet_input, t, controlnet_emb, condition, 1.0, None,None,None,None,None,False,False))
                # sum_flops += flops
                # if i == 0:
                #     sum_parms += params
                if not self.isVal:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d], dim=0) for d in down_block_res_samples]
                    mid_block_res_samples = torch.cat([torch.zeros_like(mid_block_res_samples), mid_block_res_samples], dim=0)
                
                # print(latent_model_input.shape, t.shape, text_embeddings.shape)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,
                                       down_block_additional_residuals=[d for d in down_block_res_samples],
                                       mid_block_additional_residual=mid_block_res_samples ,
                                    #    attention_mask=attention_mask,
                                       ).sample
                # flops, params = profile(self.unet, inputs=(latent_model_input, t, text_embeddings, None, None, None, None, None,
                #                        [d for d in down_block_res_samples],
                #                        mid_block_res_samples))
                # sum_flops += flops
                # if i == 0:
                #     sum_parms += params
                # cv2.imwrite('before_noise.jpg', tensor2im(self.decode_latents(start_latents))[:, :, ::-1])
                # cv2.imwrite('before.jpg', tensor2im(self.decode_latents(latents))[:, :, ::-1])
                if not self.isVal:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    # print(torch.max(noise_pred_uncond - noise_pred_text))
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                latents, org_latents = self.step(noise_pred, t, latents)
                
                # if t > 500:
                #     latents = (latents * (1 - mask_resized) + latents_list[i] * mask_resized + latents) / 2
                # print(torch.max(latents))
        #         # cv2.imwrite(f'after_{i:04d}.jpg', tensor2im(img)[:, :, ::-1])

                # progress_bar.update()
        

        if self.upsampler_w is not None:
            img = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            results_w = self.upsampler_w.decode(latents / self.upsampler_w.config.scaling_factor, condition_list).sample
            results_wo = self.upsampler_wo.decode(latents / self.upsampler_wo.config.scaling_factor, condition_list).sample
            return img, (results_w, results_wo), latents
        if return_latents:
            return latents, latents_noise
        else:
            img = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            # results = self.upsampler_w.decode(latents / self.upsampler_w.config.scaling_factor, condition_list).sample
            # a2 = time.time()
            # print(a2 - a1)
            # z = self.upsampler_w.post_quant_conv(latents / self.vae.config.scaling_factor)
            # img = self.upsampler_w.decoder(z, condition_list)
            # z = self.vae.post_quant_conv(latents / self.vae.config.scaling_factor)
            # flops, params = profile(self.vae.decoder, inputs=(z.unsqueeze(0)))
            # flops, params = profile(self.upsampler_w.decoder, inputs=(z, condition_list))
            # sum_flops += flops
            # sum_parms += params
            # print(sum_flops, sum_parms)
            # exit()
            # exit()

            # mask = (condition_images[:, :1, :, :] > 0.8 * 255) | (condition_images[:, 1:2, :, :] > 0.8 * 255) | (condition_images[:, 2:, :, :] > 0.8 * 255)
            # mask |= (condition_images[:, :1, :, :] < 0.2 * 255) | (condition_images[:, 1:2, :, :] < 0.2 * 255) | (condition_images[:, 2:, :, :] < 0.2 * 255)
            # mask_3 = mask.repeat(1, 3, 1, 1)

            # gray_src = rgb_to_grayscale(condition_images).detach()
            # gray_dst = rgb_to_grayscale(img).detach()
            # LDR_new = condition_images / gray_src * gray_dst
            # LDR_new[mask_3] = img[mask_3]
            # img = LDR_new
            
            # results = self.upsampler(condition_list, latents, img)#torch.cat([img, mask], dim=1))
            return img, None, latents