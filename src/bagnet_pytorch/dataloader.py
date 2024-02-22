import os

import PIL
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from config import (
    dataset_config,
    DATASETS,
    test_batch_size,
    train_batch_size,
)

from sklearn.model_selection import train_test_split
from constants import BODMAS_LABEL_MAPPINGS
from torch.utils.data import Dataset

# load the data
data_dir = dataset_config["data_dir"]
metadata_file = dataset_config["metadata_file"]
images_dir = dataset_config["images_dir"]
img_shape = dataset_config["img_shape"]
mean = dataset_config["mean"]
std = dataset_config["std"]



dataset = pd.read_csv(os.path.join(data_dir, metadata_file), delimiter=",")


# filtering out families with less than 50 samples:
family_counts = dataset['family'].value_counts().to_frame()
family_counts = family_counts[family_counts['count']<100]
print(len(family_counts))
rare_families = family_counts.index.values.tolist()
dataset_filtered = dataset[~dataset['family'].isin(rare_families)]
labels_filtered = dataset_filtered["family"]

ds_train, ds_valid = train_test_split(dataset_filtered, stratify=labels_filtered, test_size=0.25)


class BodmasDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform
        self.label_transform=lambda data: torch.tensor(data, dtype=torch.int),
        self.class2index = BODMAS_LABEL_MAPPINGS

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        line = self.df.iloc[index]
        filename = line["md5"]
        label = self.class2index[line["family"]]

        #TODO: read with opencv instead? - depends on the augm library
        image = PIL.Image.open(os.path.join(data_dir, images_dir, filename + '_False_False.png'))
        if self.transform is not None:
            image = self.transform(image)
        
        label = torch.tensor(label)

        return image, label



# train set
train_dataset = BodmasDataset(  # for RoadType and RoadCondition
    df = ds_train,
    transform=transforms.Compose([transforms.CenterCrop(300), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]),
)


train_sampler = torch.utils.data.RandomSampler(
    train_dataset,
    replacement=False,
    num_samples=100,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    # when running with a subset of the dataset, set Shuffle to False, otherwise to True
    shuffle=True,
    # sampler=train_sampler,  # OPTIONAL; for running with a subset of the whole dataset
)


# validation set
valid_dataset = BodmasDataset(
    df = ds_valid,
    # remove cropping - I added it so that the images fit in GPU memory; shouldn't be necessary for images equal to or smaller in size than 300x300
    transform=transforms.Compose([transforms.CenterCrop(300), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]), 
)

valid_sampler = torch.utils.data.RandomSampler(
    valid_dataset,
    replacement=False,
    num_samples=200,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    # add to work with only a random subset of the original dataset
    # sampler=valid_sampler,
)
