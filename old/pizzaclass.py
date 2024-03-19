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

class2index = {"Pepperoni":0,
                            "Bacon":1,
                            "Mushrooms":2,
                            "Onions":3,
                            "Peppers":4,
                            "Black olives":5,
                            "Tomatoes":6,
                            "Spinach":7,
                            "Fresh basil":8,
                            "Arugula":9,
                            "Broccoli":10,
                            "Corn":11,
                            "Pineapple":12}
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

        image = self.transform(image)

        return image, label


trainTransform = T.Compose([
    T.ToTensor(),
    T.Resize((384, 384), antialias=True)
])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)

        self.bnn1 = nn.BatchNorm2d(16)
        self.bnn2 = nn.BatchNorm2d(32)
        self.bnn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.LeakyReLU()

        self.fc1 = nn.Linear(135424, 100)
        self.fc2 = nn.Linear(100, 13)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
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

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    aaa = CustomDataset("pizzaGANdata/imageLabels.txt", "pizzaGANdata/images", trainTransform)

    train_size = int(0.8 * len(aaa))
    test_size = len(aaa) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(aaa, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net()
    model.to(device)

    loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    writer = SummaryWriter(log_dir="plots/classifier")
    for epoch in range(20):
        torch.cuda.empty_cache()
        train_loss = 0
        train_epoch_steps = 0
        train_size = len(train_loader.dataset)
        model.train()
        for images, labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            ypred = model(images)
            l = loss(ypred, labels)
            l.backward()
            opt.step()

            train_loss += l.item()
            train_epoch_steps += 1

        model.eval()

        train_epoch_loss = train_loss / train_epoch_steps
        train_losses.append(train_epoch_loss)
        writer.add_scalar("trainloss", train_epoch_loss, epoch)
        print(f'epoch {epoch + 1} train loss: {train_epoch_loss}\n')

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
        test_epoch_loss = test_loss / test_epoch_steps
        test_losses.append(test_epoch_loss)
        writer.add_scalar("testloss", test_epoch_loss, epoch)
        print(f'epoch {epoch + 1} test loss: {test_epoch_loss}\n')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
        }
        torch.save(state, "classifier.pth")

