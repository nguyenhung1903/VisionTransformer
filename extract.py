import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import zipfile
from PIL import Image
import os
import random
import shutil
import zipfile
from shutil import copyfile

# subset of images from train.zip
with zipfile.ZipFile('./data/train.zip', 'r') as zip_ref:
    file_list = zip_ref.namelist()
    random.shuffle(file_list)
    subset_files = file_list  # number of images as needed
    for file in subset_files:
        if 'dog' in file:
            os.makedirs('./data/train/dog', exist_ok=True)
            zip_ref.extract(file, 'data/train/dog')
        elif 'cat' in file:
            os.makedirs('./data/train/cat', exist_ok=True)
            zip_ref.extract(file, 'data/train/cat')

# subset of images from test1.zip
with zipfile.ZipFile('./data/test1.zip', 'r') as zip_ref:
    file_list = zip_ref.namelist()
    random.shuffle(file_list)
    subset_files = file_list  # number of images
    zip_ref.extractall('./data/test', subset_files)


os.system('rm -rf ./data/train.zip')
os.system('rm -rf ./data/test1.zip')
os.system('rm sampleSubmission.csv')
os.system("mv ./data/test/test1/*.* ./data/test/ && 'rm -rf ./data/test/test1")