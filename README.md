### CZ4042
This project is a modification of the original [MMALNet](https://arxiv.org/abs/2003.09150) by Fan Zhang, Meng Li, Guisheng Zhai, and Yizhao Liu.
The experiment is performed on the CompCars dataset.

### Instructions for running MMALNet for Fine Tuning
1. Download the pretrained model of ResNet-50 from 'https://download.pytorch.org/models/resnet50-19c8e357.pth' and move it to models/pretrained
2. For cases where GPU memory is insufficient: Tweak N_list, or lower batch_size, these parameters can be found in `config.py`
- The parameter N_list is N1, N2, N3 in the original paper https://arxiv.org/pdf/2003.09150v3.pdf and you can adjust them according to GPU memory
- batch_size can be lowered to 3 or below
3. Add the image folder from the unzipped CompCars data: Move `CompCars/data/image/` into `datasets/CompCars`, in the same directory as `prepare-data_csv.ipynb`
4. Download dependencies
- Create a conda environment: `conda create --name ENV_NAME`
- Ensure conda environment is activated: `conda activate ENV_NAME`
- Install Python 3.7.x: `conda install python=3.7`
- Install PyTorch and TorchVision via pytorch channel: `conda install pytorch torchvision -c pytorch`
- Install other packages: `conda install numpy pandas tqdm pillow imageio scikit-image`
- Install Tensorboard: `conda install tensorboardx -c conda-forge` 
- Install OpenCV: `conda install opencv`
6. Run `python train.py` to commence training
- During training, the log file and checkpoint file will be saved in model_path directory
7. Run `python test.py` to test the model