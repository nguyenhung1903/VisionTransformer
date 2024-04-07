# Implementation of Vision Transformer Architecture
The Vision Transformer (ViT) is a recent architecture that applies transformers, which were originally designed for natural language processing tasks, to computer vision tasks. This architecture has shown impressive results, often outperforming traditional convolutional neural networks (CNNs) on a variety of tasks.

In this repository, we provide an implementation of the ViT architecture. We aim to make this powerful model accessible and easy to use for the community. The code is written in Python and uses the PyTorch library for the implementation of the model.

This implementation includes features such as:

- Pre-training on large-scale image datasets
- Fine-tuning on specific vision tasks
- Various transformer configurations
- Visualization tools for transformer attention maps

## How to train the ViT model?
Before, training the ViT model, you need to install the requirement packages `pip install -r requirements.txt` and change the configuration file `configs/ViT_config.py`.
```bash
ViT_config = {
    "in_channels": 3,
    "num_classes": 2,
    "embed_dim": 256,
    "patch_size": 16,
    "image_size": 224,
    "num_blocks": 3,
    "nb_head": 8,
    "hidden_dim": 256,
    "dropout": 0.1,
}
```

After that, you can train the model with your datasets.
```bash
$ python train.py -h 
usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--data_dir DATA_DIR] [--pretrain_model_path PRETRAIN_MODEL_PATH] [--save_model_path SAVE_MODEL_PATH] [--ratio RATIO]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs
  --batch_size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --data_dir DATA_DIR   Data directory
  --pretrain_model_path PRETRAIN_MODEL_PATH
                        Model path
  --save_model_path SAVE_MODEL_PATH
                        Model path
  --ratio RATIO         Train/Validation ratio
```

# Results
* Model was trained for 30 epochs with a batch size of 256
* Training accuracy: 74.76%
* Validation accuracy: 73.20%
* Training loss: 0.5034
* Validation loss: 0.5293

```
$ python train.py --epochs 30 --batch_size 256 --lr 0.0001 --data_dir ./data
```
```
----------
100% 79/79 [01:44<00:00,  1.33s/it]
[TRAINING] Loss: 0.5196 Acc: 0.7352
100% 20/20 [00:18<00:00,  1.08it/s]
[VALIDATE] Loss: 0.5527 Val Acc: 0.7226
Epoch 24/29
----------
100% 79/79 [01:45<00:00,  1.33s/it]
[TRAINING] Loss: 0.5183 Acc: 0.7375
100% 20/20 [00:18<00:00,  1.08it/s]
[VALIDATE] Loss: 0.5201 Val Acc: 0.7374
Epoch 25/29
----------
100% 79/79 [01:46<00:00,  1.34s/it]
[TRAINING] Loss: 0.5192 Acc: 0.7396
100% 20/20 [00:18<00:00,  1.06it/s]
[VALIDATE] Loss: 0.5132 Val Acc: 0.7432
Epoch 26/29
----------
100% 79/79 [01:44<00:00,  1.33s/it]
[TRAINING] Loss: 0.5178 Acc: 0.7412
100% 20/20 [00:19<00:00,  1.02it/s]
[VALIDATE] Loss: 0.5518 Val Acc: 0.7108
Epoch 27/29
----------
100% 79/79 [01:45<00:00,  1.33s/it]
[TRAINING] Loss: 0.5120 Acc: 0.7453
100% 20/20 [00:18<00:00,  1.06it/s]
[VALIDATE] Loss: 0.5253 Val Acc: 0.7326
Epoch 28/29
----------
100% 79/79 [01:45<00:00,  1.34s/it]
[TRAINING] Loss: 0.5097 Acc: 0.7464
100% 20/20 [00:18<00:00,  1.08it/s]
[VALIDATE] Loss: 0.5294 Val Acc: 0.7318
Epoch 29/29
----------
100% 79/79 [01:45<00:00,  1.33s/it]
[TRAINING] Loss: 0.5034 Acc: 0.7476
100% 20/20 [00:19<00:00,  1.02it/s]
[VALIDATE] Loss: 0.5293 Val Acc: 0.7320
Saving model
```

Logs file:
```
INFO:root:Epoch: 0 - Train Loss: 0.6860 - Train Acc: 0.5553 - Val Loss: 0.6714 - Val Acc: 0.5690
INFO:root:Epoch: 1 - Train Loss: 0.6438 - Train Acc: 0.6261 - Val Loss: 0.6421 - Val Acc: 0.6390
INFO:root:Epoch: 2 - Train Loss: 0.6186 - Train Acc: 0.6581 - Val Loss: 0.6675 - Val Acc: 0.6238
INFO:root:Epoch: 3 - Train Loss: 0.5955 - Train Acc: 0.6823 - Val Loss: 0.5894 - Val Acc: 0.6856
INFO:root:Epoch: 4 - Train Loss: 0.5809 - Train Acc: 0.6944 - Val Loss: 0.5801 - Val Acc: 0.6944
INFO:root:Epoch: 5 - Train Loss: 0.5684 - Train Acc: 0.7016 - Val Loss: 0.5769 - Val Acc: 0.6910
INFO:root:Epoch: 6 - Train Loss: 0.5658 - Train Acc: 0.7027 - Val Loss: 0.5752 - Val Acc: 0.6992
INFO:root:Epoch: 7 - Train Loss: 0.5571 - Train Acc: 0.7094 - Val Loss: 0.5538 - Val Acc: 0.7114
INFO:root:Epoch: 8 - Train Loss: 0.5526 - Train Acc: 0.7131 - Val Loss: 0.5715 - Val Acc: 0.7036
INFO:root:Epoch: 9 - Train Loss: 0.5487 - Train Acc: 0.7171 - Val Loss: 0.5496 - Val Acc: 0.7132
INFO:root:Epoch: 10 - Train Loss: 0.5473 - Train Acc: 0.7156 - Val Loss: 0.5574 - Val Acc: 0.7118
INFO:root:Epoch: 11 - Train Loss: 0.5456 - Train Acc: 0.7198 - Val Loss: 0.5527 - Val Acc: 0.7116
INFO:root:Epoch: 12 - Train Loss: 0.5387 - Train Acc: 0.7215 - Val Loss: 0.5463 - Val Acc: 0.7170
INFO:root:Epoch: 13 - Train Loss: 0.5333 - Train Acc: 0.7299 - Val Loss: 0.5697 - Val Acc: 0.7012
INFO:root:Epoch: 14 - Train Loss: 0.5328 - Train Acc: 0.7254 - Val Loss: 0.5438 - Val Acc: 0.7232
INFO:root:Epoch: 15 - Train Loss: 0.5280 - Train Acc: 0.7356 - Val Loss: 0.5531 - Val Acc: 0.7184
INFO:root:Epoch: 16 - Train Loss: 0.5317 - Train Acc: 0.7275 - Val Loss: 0.5398 - Val Acc: 0.7228
INFO:root:Epoch: 17 - Train Loss: 0.5257 - Train Acc: 0.7298 - Val Loss: 0.5430 - Val Acc: 0.7232
INFO:root:Epoch: 18 - Train Loss: 0.5269 - Train Acc: 0.7360 - Val Loss: 0.5294 - Val Acc: 0.7312
INFO:root:Epoch: 19 - Train Loss: 0.5183 - Train Acc: 0.7423 - Val Loss: 0.5303 - Val Acc: 0.7344
INFO:root:Epoch: 20 - Train Loss: 0.5236 - Train Acc: 0.7329 - Val Loss: 0.5264 - Val Acc: 0.7302
INFO:root:Epoch: 21 - Train Loss: 0.5170 - Train Acc: 0.7408 - Val Loss: 0.5419 - Val Acc: 0.7224
INFO:root:Epoch: 22 - Train Loss: 0.5180 - Train Acc: 0.7410 - Val Loss: 0.5366 - Val Acc: 0.7268
INFO:root:Epoch: 23 - Train Loss: 0.5196 - Train Acc: 0.7352 - Val Loss: 0.5527 - Val Acc: 0.7226
INFO:root:Epoch: 24 - Train Loss: 0.5183 - Train Acc: 0.7375 - Val Loss: 0.5201 - Val Acc: 0.7374
INFO:root:Epoch: 25 - Train Loss: 0.5192 - Train Acc: 0.7396 - Val Loss: 0.5132 - Val Acc: 0.7432
INFO:root:Epoch: 26 - Train Loss: 0.5178 - Train Acc: 0.7412 - Val Loss: 0.5518 - Val Acc: 0.7108
INFO:root:Epoch: 27 - Train Loss: 0.5120 - Train Acc: 0.7453 - Val Loss: 0.5253 - Val Acc: 0.7326
INFO:root:Epoch: 28 - Train Loss: 0.5097 - Train Acc: 0.7464 - Val Loss: 0.5294 - Val Acc: 0.7318
INFO:root:Epoch: 29 - Train Loss: 0.5034 - Train Acc: 0.7476 - Val Loss: 0.5293 - Val Acc: 0.7320
```