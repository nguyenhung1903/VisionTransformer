import torch
import torch.nn as nn

def predict(model, img, classnames, device, data_transforms):
    pred = model(data_transforms['train'](img).unsqueeze(0).to(device))
    score = nn.functional.softmax(pred, dim=1)[0]
    label = torch.argmax(score)
    return classnames[label], score[label].item()