# Event-guided HDR Reconstruction with Diffusion Prior

Yixin Yang, Jiawei Zhang, Yang Zhang, Yunxuan Wei, Dongqing Zou, Jimmy Ren, and Boxin Shi.

## Requirements

- Python 3.10
- CUDA-compatible GPU

## Installation

### 1. Create conda environment

```bash
conda create -n hdrev-diffu python=3.10
conda activate hdrev-diffu
```

### 2. Install PyTorch

```bash
pip3 install torch torchvision torchaudio
```

### 3. Install other dependencies

```bash
pip install lpips wandb diffusers omegaconf transformers opencv-python h5py hdf5plugin matlablib
```

## Dataset

Please downloading our training dataset (https://www.modelscope.cn/datasets/Mcallor/HDR-EVS/) and specify the dataroot in configuration file.

## Usage

### Training

#### Main Model Training

```bash
python train.py \
    --name "experiment_name" \
    --config "config/Train_stage1+2.yaml" \
```

#### Upsampler Training

Inference sampling results without refinement(change the dataroot in config file to match the training data):
```bash
python test.py \
    --exp_name "test_experiment" \
    --config "config/Test.yaml" \
    --pretrained_unet "path/to/trained/model.ckpt" \
    --save
```

Start training based on the saved latents (change the dataroot in config file to match the saved latents):
```bash
python train_upsampler.py \
    --name "upsampler_experiment" \
    --config "config/Train_upsampler.yaml" \
    --pretrained_unet "path/to/trained/model.ckpt" \
```

### Testing
Download checkpoint: [GoogleDrive](https://drive.google.com/drive/folders/1Jmjkc4itLE-QagcCEpnCGtvh-z5QUlfz?usp=share_link)
```bash
python test.py \
    --exp_name "test_experiment" \
    --config "config/Test.yaml" \
    --pretrained_unet "path/to/trained/model.ckpt" \
    --pretrained_upsample "path/to/trained/upsampler.ckpt" \
```

## Configuration

### Main Configuration Parameters

- `output_dir`: Output directory
- `pretrained_model_path`: Path to pre-trained Stable Diffusion model
- `pretrained_controlnet_model_path`: Path to pre-trained ControlNet model
- `conditioning_channels`: Conditioning channel configuration
- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `max_train_steps`: Maximum training steps

### Dataset Configuration

- `dataset_type`: Dataset type (evs_image_h5single)
- `dataroot`: Data root directory
- `patch_size`: Image patch size
- `num_bins`: Number of event time bins
- `event_representation`: Event representation method

## Model Architecture

### HDRev_Encoder
Specially designed event data encoder for processing temporal information from event cameras.

### OursControlNetModel
ControlNet-based conditional control model supporting multiple input types:
- Event data (evs)
- Low dynamic range images (ldr)
- Event + low dynamic range images (evs+ldr)

### OursAutoencoderKL
Improved autoencoder for image upsampling and detail enhancement.

## Data Format

The project supports HDF5 format datasets containing:
- Event data (pixel_events)
- Low dynamic range images (pixel_images)
- High dynamic range images (target_images)

## Citation

```bibtex
@inproceedings{yang2025eventguided,
  author    = {Yang, Yixin and Zhang, Jiawei and Zhang, Yang and Wei, Yunxuan and Zou, Dongqing and Ren, Jimmy and Shi, Boxin},
  title     = {Event-guided HDR Reconstruction with Diffusion Priors},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
}
```

## Contact

For questions or suggestions, please contact us via:
- Email: yangyixin93@pku.edu.cn
