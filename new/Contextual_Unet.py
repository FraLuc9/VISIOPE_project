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
from contextual_loss import functional as F
from contextual_loss.modules.contextual import ContextualLoss as cl


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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #checkpoint = torch.load("latest_Unet_contextual_loss__w.pth")
    model = OliveOil()
    #model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    criterion_source = cl(band_width=0.1, use_vgg=True, vgg_layer1='relu4_2').to(device)
    criterion_target = cl(band_width=0.2, use_vgg=True, vgg_layer1='relu2_2', vgg_layer2='relu3_2', vgg_layer3='relu4_2').to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    #optimizer.load_state_dict(checkpoint["optimizer"])
    train_losses = []
    test_losses = []
    best_loss = 100
    writer = SummaryWriter(log_dir="plots/Unet_contextualLoss_multiFixed/")
    epochs = 20
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_source_loss = 0
        train_content_loss = 0
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

            l_target = criterion_target(ypred, images2)
            #l_target = criterion_t2(ypred, images2)
            #l_t2 = criterion_t2(ypred, images2)
            #l_t3 = criterion_t3(ypred, images2)
            #l_target = l_t1 + l_t2# + l_t3
            l_source = criterion_source(ypred, images)
            l = l_target + l_source
            l.backward()

            optimizer.step()
            train_source_loss += l_source.item()
            train_content_loss += l_target.item()
            train_loss += l.item()
            train_epoch_steps += 1

        model.eval()

        train_epoch_loss = train_loss / train_epoch_steps
        train_epoch_source_loss = train_source_loss / train_epoch_steps
        train_epoch_content_loss = train_content_loss / train_epoch_steps
        train_losses.append(train_epoch_loss)
        writer.add_scalar("train_source_loss", train_epoch_source_loss, epoch + 1)
        writer.add_scalar("train_content_loss", train_epoch_content_loss, epoch + 1)
        writer.add_scalar("trainloss", train_epoch_loss, epoch + 1)
        print(f'epoch {epoch + 1} train loss: {train_epoch_loss}\n')

        test_loss = 0
        test_source_loss = 0
        test_content_loss = 0
        test_epoch_steps = 0
        test_size = len(test_loader.dataset)
        with torch.no_grad():
            for images, labels, images2, labels2 in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                images2 = images2.to(device)
                labels2 = labels2.to(device)

                ypred = model(images, labels2)
                #l_t1 = criterion_t1(ypred, images2)
                l_target = criterion_target(ypred, images2)
                #l_t3 = criterion_t3(ypred, images2)
                #l_target = l_t1 + l_t2 + l_t3
                l_source = criterion_source(ypred, images)
                l = l_target + l_source
                test_source_loss += l_source.item()
                test_content_loss += l_target.item()
                test_loss += l.item()
                test_epoch_steps += 1
        test_epoch_loss = test_loss / test_epoch_steps
        test_epoch_source_loss = test_source_loss / test_epoch_steps
        test_epoch_content_loss = test_content_loss / test_epoch_steps
        test_losses.append(test_epoch_loss)
        writer.add_scalar("test_source_loss", test_epoch_source_loss, epoch + 1)
        writer.add_scalar("test_content_loss", test_epoch_content_loss, epoch + 1)
        writer.add_scalar("testloss", test_epoch_loss, epoch + 1)
        print(f'epoch {epoch + 1} test loss: {test_epoch_loss}\n')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, "latest_Unet_contextual_loss__w2.pth")
        if test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            torch.save(state, "best_Unet_contextual_loss__w2.pth")

