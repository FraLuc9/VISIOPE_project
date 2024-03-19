import torch
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch.utils.data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, labels_path, images_path, transform):
        self.df = pd.read_csv(labels_path, names=["Labels"])
        self.df = self.df.iloc[:, 0].str.split(" ")
        self.image_dir = images_path
        self.transform = transform
        images = pd.Series(os.listdir(self.image_dir))

        self.df = pd.concat([self.df, images], axis="columns")
        self.df.columns = ["Labels", "Images"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index]["Labels"]
        label = np.asarray(label, float)
        label = torch.from_numpy(label)
        filename = self.df.loc[index]["Images"]
        image = Image.open(os.path.join(self.image_dir, filename))
        image = image.convert('RGB')
        image = self.transform(image)

        random = torch.randint(low=0, high=len(self.df), size=(1,)).item()
        label2 = self.df.iloc[random]["Labels"]
        label2 = np.asarray(label2, float)
        label2 = torch.from_numpy(label2)
        while torch.equal(label, label2):
            random = torch.randint(low=0, high=len(self.df), size=(1,)).item()
            label2 = self.df.iloc[random]["Labels"]
            label2 = np.asarray(label2, float)
            label2 = torch.from_numpy(label2)
        filename2 = self.df.loc[random]["Images"]
        image2 = Image.open(os.path.join(self.image_dir, filename2))
        image2 = image2.convert('RGB')
        image2 = self.transform(image2)

        return image, label, image2, label2
