# HDRev-Diff

**HDRev-Diff** is ...

## 🧰 Requirements

To set up the environment, you can use the provided `requirements.txt` or create a Conda environment:

```bash
conda create -n hdrev-diff python=3.10
conda activate hdrev-diff
pip install -r requirements.txt
```


Alternatively, you can manually install the dependencies:

```bash
pip install torch torchvision torchaudio
pip install lpips wandb diffusers omegaconf transformers opencv-python h5py hdf5plugin matplotlib
```


## 🧪 Testing

To run the provided test script:

```bash
python test_1+2.py
```



## 🚀 Training

To train the upsampling model:

```bash
python train_upsampler.py
```



Ensure that you have prepared your dataset and configured the necessary parameters before initiating the training process.

## 📂 Directory Structure

* `config/`: Configuration files for the model and training parameters.
* `data_processing/`: Scripts for preparing and loading datasets.
* `networks/`: Implementation of the neural network architectures.
* `pipeline/`: Main training and evaluation pipeline.
* `utils/`: Utility functions and helper scripts.
* `Stage1+2.py`: Combined script for stages 1 and 2 of the process.
* `dorfCurves.txt`: Data file related to the restoration process.
* `test_1+2.py`: Test script for evaluating the model.
* `train_upsampler.py`: Script for training the upsampling model.
* `requirements.txt`: List of Python package dependencies.

## 📄 License

This project is licensed under the MIT License.
