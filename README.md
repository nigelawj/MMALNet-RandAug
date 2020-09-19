###TO TRAIN MMAL
1.download the pretrained model of ResNet-50 and move it to models/pretrained
2.if your GPU memory is not enough. 
The parameter N_list is N1, N2, N3 in the original paper https://arxiv.org/pdf/2003.09150v3.pdf and you can adjust them according to GPU memory. 
(its related to the level of importance in of parts retrived by attention) 
N1 is the number of parts that are most important, N2 is the number of parts that are 2nd most important and N3 is the number of parts that are 3rd most important. 
Ask weimin to explain if dont understand HAHA
3.add image folder from compcars data 'CompCars/data/image/' into 'datasets/ComCars/image/'
4.check if need to change anything in config. Weimin only change those related to dataset.
5.download dependencies
6.python train.py
7.During training, the log file and checkpoint file will be saved in model_path directory.

cz4042

### Preparing EfficientNetbX Weights
1. Download the appropriate weights from [official GitHub repo](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet#2-using-pretrained-efficientnet-checkpoints)
2. Unzip it to <filepath>
3. Run `python efficientnet_weight_update_util.py --model b1 --notop --ckpt <filepath> --o <any filename>`
4. Load weights


