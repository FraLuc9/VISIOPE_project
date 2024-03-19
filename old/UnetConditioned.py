import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from time import time
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform):
        self.df = pd.read_csv(csv_path, names=["labels"])
        self.df = self.df.iloc[:, 0].str.split("  ")
        imgs = pd.Series(os.listdir(images_folder))
        self.df = pd.concat([self.df, imgs], axis="columns")

        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index]["labels"]
        label = np.asarray(label, float)
        label = torch.from_numpy(label)
        filename = self.df.loc[index][0]
        image = Image.open(os.path.join(self.images_folder, filename))

        random = torch.randint(low=0, high=len(self.df), size=(1,)).item()
        label2 = self.df.iloc[random]["labels"]
        label2 = np.asarray(label2, float)
        label2 = torch.from_numpy(label2)
        while torch.equal(label, label2):
            random = torch.randint(low=0, high=len(self.df), size=(1,)).item()
            label2 = self.df.iloc[random]["labels"]
            label2 = np.asarray(label2, float)
            label2 = torch.from_numpy(label2)
        filename2 = self.df.loc[random][0]
        image2 = Image.open(os.path.join(self.images_folder, filename2))
        image2 = image2.convert("RGB")
        image2 = self.transform(image2)
        image = image.convert("RGB")
        image = self.transform(image)

        return image, label, image2, label2


trainTransform = T.Compose([
    T.ToTensor(),
    T.Resize((256, 256), antialias=True)
])

resnetTransform = T.Compose([
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnetInverseTransform = T.Compose([
    T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
               T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
               ])
])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.conv3(residual)
        out = self.relu(out)
        return out


class OliveOil(nn.Module):
    def __init__(self):
        super().__init__()

        self.inconv = nn.Conv2d(3, 32, 3, padding=1)

        self.downconv1 = ResidualBlock(32, 64)

        self.downconv2 = ResidualBlock(64, 128)

        self.downconv3 = ResidualBlock(128, 256)

        self.downconv4 = ResidualBlock(256, 512)

        self.conv51 = nn.Conv2d(2560, 1024, 3, 1, 1)
        self.conv52 = nn.Conv2d(3072, 512, 3, 1, 1)

        self.upconv1 = ResidualBlock(1024, 256)

        self.upconv2 = ResidualBlock(512, 128)

        self.upconv3 = ResidualBlock(256, 64)
        self.upconv4 = ResidualBlock(128, 32)

        self.outconv = nn.Conv2d(64, 3, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.bnn51 = nn.BatchNorm2d(1024)
        self.bnn52 = nn.BatchNorm2d(512)

        self.relu = nn.LeakyReLU()

        self.cond = nn.Embedding(13, 8)
        self.linear = nn.Linear(2048, 2048)

    def forward(self, x, features):
        features = self.linear(features)
        features = features.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        # print(features.shape)
        xin = self.inconv(x)

        # downconv1
        x1 = self.pool(xin)
        x1 = self.downconv1(x1)

        # downconv2
        x2 = self.pool(x1)
        x2 = self.downconv2(x2)

        # downconv3
        x3 = self.pool(x2)
        x3 = self.downconv3(x3)

        # downconv4
        x4 = self.pool(x3)
        x4 = self.downconv4(x4)

        # papillon
        x5 = self.pool(x4)
        x5 = self.relu(self.bnn51(self.conv51(torch.cat((x5, features), dim=1))))
        x5 = self.relu(self.bnn52(self.conv52(torch.cat((x5, features), dim=1))))

        # x4 = self.conv42(torch.cat((x4, labconv), dim=1))

        # upconv1
        x6 = self.upscale(x5)
        x6 = torch.cat((x6, x4), dim=1)
        x6 = self.upconv1(x6)

        # upconv2
        x7 = self.upscale(x6)
        x7 = torch.cat((x7, x3), dim=1)
        x7 = self.upconv2(x7)

        # upconv3
        x8 = self.upscale(x7)
        x8 = torch.cat((x8, x2), dim=1)
        x8 = self.upconv3(x8)

        # upconv4
        x9 = self.upscale(x8)
        x9 = torch.cat((x9, x1), dim=1)
        x9 = self.upconv4(x9)

        x = self.upscale(x9)
        x = torch.cat((x, xin), dim=1)
        x = self.outconv(x)

        # upconv4

        return x


if __name__ == '__main__':
    torch.manual_seed(95)
    aaa = CustomDataset("pizzaGANdata/imageLabels.txt", "pizzaGANdata/images", trainTransform)

    train_size = int(0.8 * len(aaa))
    test_size = len(aaa) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(aaa, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model = OliveOil()
    resNet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights="DEFAULT")
    resNetWeights = torch.load("best_resnet.pth")
    resNet.fc = nn.Linear(resNet.fc.in_features, 13)
    resNet.load_state_dict(resNetWeights['state_dict'])
    resNet.fc = nn.Identity()
    resNet.to(device)
    # checkpoint = torch.load("latest_autoencoderdeep.pth")
    # model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    # epoch = checkpoint['epoch']
    loss = nn.MSELoss()
    resNet.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # opt.load_state_dict(checkpoint['optimizer'])

    train_losses = []

    test_losses = []
    train_acc = []
    test_acc = []
    best_loss = 100

    epochs = 20
    writer = SummaryWriter(log_dir="plots/unetConditioned/")
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
            opt.zero_grad()

            ypred = model(images, features)
            with torch.no_grad():
                features1 = resNet(resnetTransform(ypred))
                features2 = resNet(resnetTransform(images2))

            l = loss(ypred, images) + loss(resnetInverseTransform(features1), resnetInverseTransform(features2))
            l.backward()
            opt.step()

            train_loss += l.item()
            train_epoch_steps += 1

        model.eval()

        train_epoch_loss = train_loss / train_epoch_steps

        train_losses.append(train_epoch_loss)
        writer.add_scalar("train_loss", train_epoch_loss, epoch + 1)

        test_loss = 0
        test_epoch_steps = 0
        test_size = len(test_loader.dataset)
        with torch.no_grad():
            for images, labels, images2, labels2 in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                images2 = images2.to(device)
                labels2 = labels2.to(device)
                features = resNet(images2)
                predictions = model(images, features)

                l = loss(predictions, images2)
                test_loss += l.item()
                test_epoch_steps += 1

        test_epoch_loss = test_loss / test_epoch_steps

        test_losses.append(test_epoch_loss)
        writer.add_scalar("test_loss", test_epoch_loss, epoch + 1)
        print(f'epoch {epoch + 1} train loss: {train_epoch_loss}, test loss: {test_epoch_loss}\n')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
        }
        torch.save(state, "latest_unetWConditioning.pth")
        if test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            torch.save(state, "best_unetWConditioning.pth")
