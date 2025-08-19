import torch
import inspect
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torchvision.transforms.functional as vF

class Stagepipeline(DiffusionPipeline):

    def __init__(self, vae, text_encoder, tokenizer, unet, controlnet, cond_encoder, scheduler, upsampler, upsampler_w,
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
            upsampler_w=upsampler_w
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.isVal = isVal
    
    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

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

    @torch.no_grad()
    def __call__(self, prompt, condition_images, condition_events, height, width, num_inference_steps, 
                negative_prompt=None, eta=0.0, generators=None, latents=None, guidance_scale=1.0, latents_noise=None,
                gt=None, use_image_prompt=False, prompt_image=None, return_latents=False):

        if guidance_scale < 1:
            raise ValueError(f'guidance scale should greater than one, while the input is {guidance_scale}')
        batch_size = prompt.shape[0]
        device = self.unet.device

        text_embeddings = prompt if self.isVal else torch.cat([prompt, prompt])

        condition_images = condition_images.to(device)
        condition_events = condition_events.to(device)

        conditions, condition_list = self.cond_encoder(condition_images, condition_events)

        # prepare timestamps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timestamps = self.scheduler.timesteps

        # prepare latents
        num_channels_latents = self.vae.config.latent_channels
        if latents is None:
            latents, noise = self.prepare_latents(batch_size, num_channels_latents, height, width, device, generators)
        
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
                if not self.isVal:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d], dim=0) for d in down_block_res_samples]
                    mid_block_res_samples = torch.cat([torch.zeros_like(mid_block_res_samples), mid_block_res_samples], dim=0)
                
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,
                                       down_block_additional_residuals=[d for d in down_block_res_samples],
                                       mid_block_additional_residual=mid_block_res_samples ,
                                       ).sample
                if not self.isVal:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents, org_latents = self.step(noise_pred, t, latents)
                
        if return_latents:
            return latents, latents_noise
        else:
            img = self.upsampler.decode(latents / self.vae.config.scaling_factor).sample
            if self.upsampler_w != None:
                results = self.upsampler_w.decode(latents / self.upsampler_w.config.scaling_factor, condition_list).sample
            else: 
                results = None
            return img, results, latents