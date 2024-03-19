import torch
from torch import nn
import numpy as np
import pandas as pd
import os
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
        image = image.convert("RGB")
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
        image = self.transform(image)

        return image, label, image2, label2


trainTransform = T.Compose([
    T.ToTensor(),
    T.Resize((256, 256), antialias=True)
])


class OliveOil(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     nn.MaxPool2d(2, 2, ceil_mode=True),
        #     nn.Conv2d(3, 32, 3),
        #     nn.ELU(),
        #
        # )
        self.inconv = nn.Conv2d(3, 32, 7, 1, 3)

        self.conv1 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(269, 512, 3, 1, 1)

        self.deconv1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.deconv2 = nn.Conv2d(269, 128, 3, 1, 1)
        self.deconv3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.deconv4 = nn.Conv2d(64, 32, 5, 1, 2)
        self.outconv = nn.Conv2d(32, 3, 7, 1, 3)

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.unpool = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.act = nn.ELU()

        self.bnn0 = nn.BatchNorm2d(32)
        self.bnn1 = nn.BatchNorm2d(64)
        self.bnn2 = nn.BatchNorm2d(128)
        self.bnn3 = nn.BatchNorm2d(256)
        self.bnn4 = nn.BatchNorm2d(512)

        self.cond = nn.Linear(13, 256)

        self.decond1 = nn.Embedding(13, 16)
        self.decond2 = nn.Embedding(13, 64)

    def forward(self, x, labels):

        labconv = self.decond1(labels.long()).unsqueeze(3).repeat(1, 1, 1, 16)
        labdeconv = self.decond2(labels.long()).unsqueeze(3).repeat(1, 1, 1, 64)

        x = self.inconv(x)

        x = self.pool(x)

        x = self.conv1(x)
        x = self.bnn1(x)
        x = self.act(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.bnn2(x)
        x = self.act(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.bnn3(x)
        x = self.act(x)

        x = self.pool(x)

        x = self.conv4(torch.cat((x, labconv), dim=1))
        x = self.bnn4(x)
        x = self.act(x)

        x = self.unpool(x)

        x = self.deconv1(x)#+ self.decond1(labels.float()))
        x = self.bnn3(x)
        x = self.act(x)

        x = self.unpool(x)

        x = self.deconv2(torch.cat((x, labdeconv), dim=1))
        x = self.bnn2(x)
        x = self.act(x)

        x = self.unpool(x)

        x = self.deconv3(x)
        x = self.bnn1(x)
        x = self.act(x)

        x = self.unpool(x)

        x = self.deconv4(x)
        x = self.bnn0(x)
        x = self.act(x)

        x = self.outconv(x)

        return x


if __name__ == '__main__':
    aaa = CustomDataset("pizzaGANdata/imageLabels.txt", "pizzaGANdata/images", trainTransform)

    train_size = int(0.8 * len(aaa))
    test_size = len(aaa) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(aaa, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)#, num_workers=6)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = OliveOil()
    checkpoint = torch.load("latest_autoencoderdeep.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    epoch = checkpoint['epoch']
    loss = nn.MSELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.load_state_dict(checkpoint['optimizer'])
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    best_loss = 100
    writer = SummaryWriter(log_dir="plots/autoencoder/")
    for e in range(20):
        epoch = epoch + 1
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
            ypred = model(images, labels2)
            l = loss(ypred, images2)
            l.backward()
            opt.step()

            train_loss += l.item()
            train_epoch_steps += 1

        model.eval()

        train_epoch_loss = train_loss / train_epoch_steps
        train_losses.append(train_epoch_loss)
        writer.add_scalar("train_loss", train_epoch_loss, epoch + 1)
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

                l = loss(predictions, images2)

                test_loss += l.item()
                test_epoch_steps += 1
        test_epoch_loss = test_loss / test_epoch_steps

        test_losses.append(test_epoch_loss)
        writer.add_scalar("test_loss", test_epoch_loss, epoch + 1)
        print(f'epoch {epoch + 1} test loss: {test_epoch_loss}\n')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
        }
        torch.save(state, "latest_autoencoderdeepest.pth")
        if test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            torch.save(state, "best_autoencoderdeepest.pth")

