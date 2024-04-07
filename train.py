import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import  datasets
from tqdm import tqdm
import numpy as np
import argparse
from models import ViT
from configs import ViT_config
import torchsummary as summary
from modules import data_transforms

import datetime
import logging

if not os.path.exists("logs"):
    os.makedirs("logs")

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename=f"logs/{timestamp}.txt", filemode='a',level=logging.INFO)

def validate_model(model, criterion):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloaders['val']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects.double() / dataset_sizes['val']

    print('[VALIDATE] Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    return epoch_acc, epoch_loss
    

def train_model(model, criterion, optimizer, num_epochs=25, save_model_path=None):
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}
    max_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        # use tqdm for progress bar
        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        
        loss_hist['train'].append(epoch_loss)
        acc_hist['train'].append(epoch_acc)
        print('[TRAINING] Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        valid_epoch_acc, valid_epoch_loss = validate_model(model, criterion)
        
        if valid_epoch_acc > max_acc:
            print(f"Saving best model at {save_model_path}/best.pth")
            torch.save(model.state_dict(), f"{save_model_path}/best.pth")
            max_acc = valid_epoch_acc
            
        torch.save(model.state_dict(), f"{save_model_path}/checkpoint-epoch-{epoch}.pth")
            
        loss_hist['val'].append(valid_epoch_loss)
        acc_hist['val'].append(valid_epoch_acc)
        
        logging.info(f"Epoch: {epoch} - Train Loss: {epoch_loss} - Train Acc: {epoch_acc} - Val Loss: {valid_epoch_loss} - Val Acc: {valid_epoch_acc}")
        
    return model, (loss_hist, acc_hist)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    args.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    args.add_argument("--pretrain_model_path", type=str, default="./saved_models/best.pth", help="Model path")
    args.add_argument("--save_model_path", type=str, default="./saved_models", help="Model path")
    args.add_argument("--ratio", type=float, default=0.8, help="Train/Validation ratio")
    args.add_argument("--number_workers", type=int, default=4, help="Number of workers")
    
    opt = args.parse_args()
    print(opt)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViT(**ViT_config).to(device)
    summary.summary(model, (3, 224, 224))
    
    if os.path.exists(opt.pretrain_model_path):
        model.load_state_dict(torch.load(opt.pretrain_model_path))
        print("Pretrained model loaded")
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(opt.data_dir, x), data_transforms[x]) for x in ['train']}
    # split the data into training and validation sets
    train_size = int(opt.ratio * len(image_datasets['train']))
    val_size = len(image_datasets['train']) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(image_datasets['train'], [train_size, val_size])

    # create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.number_workers),
        'val': DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.number_workers)
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    # Create save model path
    if (not os.path.exists(opt.save_model_path)):
        os.makedirs(opt.save_model_path)
    
    print("Training model")
    model, (loss_hist, acc_hist) = train_model(model, criterion, optimizer, num_epochs=opt.epochs, save_model_path=opt.save_model_path)
    
    print("Saving model")
    torch.save(model.state_dict(), f"{opt.save_model_path}/last.pth")