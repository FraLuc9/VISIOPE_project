import pandas as pd
import torch
import numpy as np
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torcheval.metrics as metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Dataset_IO import CustomDataset
from Unet_model import OliveOil

from time import time


trainTransform = T.Compose([
    T.ToTensor(),
    T.Resize((256, 256), antialias=True),
])
resnetTransform = T.Compose([
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == '__main__':

    torch.manual_seed(95)
    pizzaset = CustomDataset("Pizza10/imageLabels.txt", "Pizza10/images", trainTransform)

    train_size = int(0.8 * len(pizzaset))
    test_size = len(pizzaset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(pizzaset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights="DEFAULT")
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    checkpoint = torch.load("best_resnet_multiloss.pth")
    resnet.load_state_dict(checkpoint["state_dict"])
    resnet.fc = nn.Identity()
    resnet.to(device)
    resnet.eval()

    model = OliveOil()
    checkpt = torch.load("latest_Unet_resnet_loss.pth")
    model.load_state_dict(checkpt["state_dict"])
    model.to(device)

    loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_losses = []
    test_losses = []
    best_loss = 100
    writer = SummaryWriter(log_dir="plots/Unet_resnetLoss/")
    epochs = 50
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss = 0
        train_epoch_steps = 0
        train_size = len(train_loader.dataset)
        model.train()
        for images, labels, images2, labels2 in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            images2 = images2.to(device)
            labels2 = labels2.to(device)

            optimizer.zero_grad()

            ypred = model(images, labels2)

            feat1 = resnet(resnetTransform(ypred))
            feat2 = resnet(resnetTransform(images2))

            l = loss(feat1, feat2)
            l.backward()

            optimizer.step()

            train_loss += l.item()
            train_epoch_steps += 1

        model.eval()

        train_epoch_loss = train_loss / train_epoch_steps
        train_losses.append(train_epoch_loss)
        writer.add_scalar("trainloss", train_epoch_loss, epoch + 1)
        print(f'epoch {epoch + 1} train loss: {train_epoch_loss}\n')

        test_loss = 0
        test_epoch_steps = 0
        test_size = len(test_loader.dataset)
        with torch.no_grad():
            for images, labels, images2, labels2 in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                images2 = images2.to(device)
                labels2 = labels2.to(device)

                predictions = model(images, labels2)
                feat1 = resnet(predictions)
                feat2 = resnet(images2)

                l = loss(feat1, feat2)

                test_loss += l.item()
                test_epoch_steps += 1
        test_epoch_loss = test_loss / test_epoch_steps
        test_losses.append(test_epoch_loss)
        writer.add_scalar("testloss", test_epoch_loss, epoch + 1)
        print(f'epoch {epoch + 1} test loss: {test_epoch_loss}\n')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, "latest_Unet_resnet_loss__.pth")
        if test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            torch.save(state, "best_Unet_resnet_loss__.pth")

