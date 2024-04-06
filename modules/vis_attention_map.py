import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .data_transforms import data_transforms


def get_attention_map(model, img, device):
    x = model.input_layer(data_transforms['test'](img).unsqueeze(0).to(device))
    x = model.encoder(x)
    return x.squeeze(0).detach().cpu().numpy()
