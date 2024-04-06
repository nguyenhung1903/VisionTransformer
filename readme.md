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