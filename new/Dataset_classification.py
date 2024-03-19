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

        return image, label
