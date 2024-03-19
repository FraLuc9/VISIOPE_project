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
import VGG_loss

trainTransform = T.Compose([
    T.ToTensor(),
    T.Resize((128, 128), antialias=True),
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

    model = OliveOil()
    model.to(device)
    crit = VGG_loss.WeightedLoss([VGG_loss.VGGLoss(shift=2),
                                  #nn.MSELoss(),
                                  VGG_loss.TVLoss(p=1)],
                                 [1, 1]).to(device)
    loss2 = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_losses = []
    test_losses = []
    best_loss = 100
    writer = SummaryWriter(log_dir="plots/Unet_VGGLoss/")
    epochs = 50

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss = 0
        train_epoch_steps = 0
        train_size = len(train_loader.dataset)
        model.train()
        crit.train()
        for images, labels, images2, labels2 in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            images2 = images2.to(device)
            labels2 = labels2.to(device)

            optimizer.zero_grad()

            ypred = model(images, labels2)
            l1 = crit(ypred, images2)
            l2 = loss2(ypred, images)
            loss = l1 + l2
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_epoch_steps += 1

        model.eval()
        crit.eval()
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
                l1 = crit(predictions, images2)
                l2 = loss2(predictions, images)
                loss = l1 + l2
                test_loss += loss.item()
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
        torch.save(state, "latest_Unet_vgg_loss.pth")
        if test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            torch.save(state, "best_Unet_vgg_loss__.pth")