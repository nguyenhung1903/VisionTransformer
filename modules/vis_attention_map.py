import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .data_transforms import data_transforms

def get_attention_map(model, img, device, ViT_config):
    head_dim = ViT_config["embed_dim"] // ViT_config["nb_head"]
    x = model.input_layer(data_transforms['train'](img).unsqueeze(0).to(device))
    x = model.encoder[0].msa.ln1(x)
    batch_size, nb_patch, _ = x.size()
    q = model.encoder[0].msa.w_q(x)
    q = q.view(batch_size, ViT_config["nb_head"], nb_patch, head_dim)
    k = model.encoder[0].msa.w_k(x)
    k = k.view(batch_size, ViT_config["nb_head"], nb_patch, head_dim)
    v = model.encoder[0].msa.w_v(x)
    v = v.view(batch_size, ViT_config["nb_head"], nb_patch, head_dim)

    # inner product
    # (B, nb_head, N, D//nb_head) × (B, nb_head, D//nb_head, N) -> (B, nb_head, N, N)
    dots = (q @ k.transpose(2, 3)) / head_dim **0.5
    # softmax by columns
    # dim=3 eq dim=-1. dim=-1 applies softmax to the last dimension
    attn = torch.nn.functional.softmax(dots, dim=3)
    return attn.squeeze(0).mean(0).detach().cpu().numpy()