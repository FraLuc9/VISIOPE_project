import pandas as pd
import torch
import numpy as np
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torcheval.metrics as metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Dataset_classification import CustomDataset


trainTransform = T.Compose([
    T.ToTensor(),
    T.Resize((256, 256), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights="DEFAULT")
    freeze = True
    freeze_until = "layer4"
    for name, param in model.named_parameters():
        if freeze_until in name:
            freeze = False
        if freeze:
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
 
    loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    best_loss = 100
    writer = SummaryWriter(log_dir="plots/pizzaresnet/")
    accuracy_metric = metrics.MultilabelAccuracy(criteria="hamming")
    epochs = 2
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss = 0
        train_epoch_steps = 0
        train_size = len(train_loader.dataset)
        model.train()
        for images, labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            ypred = model(images)
            l = loss(ypred, labels)
            l.backward()
            optimizer.step()

            train_loss += l.item()
            train_epoch_steps += 1
            accuracy_metric.update(ypred, labels)

        train_epoch_acc = accuracy_metric.compute()
        train_acc.append(train_epoch_acc)
        accuracy_metric.reset()
        model.eval()

        train_epoch_loss = train_loss / train_epoch_steps
        train_losses.append(train_epoch_loss)
        writer.add_scalar("trainloss", train_epoch_loss, epoch + 1)
        writer.add_scalar("trainaccuracy", train_epoch_acc, epoch + 1)
        print(f'epoch {epoch + 1} train accuracy: {train_acc[-1]}, train loss: {train_epoch_loss}\n')

        test_loss = 0
        test_epoch_steps = 0
        test_size = len(test_loader.dataset)
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)

                l = loss(predictions, labels)

                test_loss += l.item()
                test_epoch_steps += 1
                accuracy_metric.update(predictions, labels)
        test_epoch_loss = test_loss / test_epoch_steps
        test_epoch_acc = accuracy_metric.compute()
        test_acc.append(test_epoch_acc)
        accuracy_metric.reset()
        test_losses.append(test_epoch_loss)
        writer.add_scalar("testloss", test_epoch_loss, epoch + 1)
        writer.add_scalar("testaccuracy", test_epoch_acc, epoch + 1)
        print(f'epoch {epoch + 1} test accuracy: {test_acc[-1]}, test loss: {test_epoch_loss}\n')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, "latest_resnet_multiloss.pth")
        if test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            torch.save(state, "best_resnet_multiloss.pth")
