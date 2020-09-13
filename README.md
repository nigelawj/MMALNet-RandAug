cz4042

### Preparing EfficientNetbX Weights
1. Download the appropriate weights from [official GitHub repo](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet#2-using-pretrained-efficientnet-checkpoints)
2. Unzip it to <filepath>
3. Run `python efficientnet_weight_update_util.py --model b1 --notop --ckpt <filepath> --o <any filename>`
4. Load weights
