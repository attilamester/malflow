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

def set_to_dict(s):
    keys = list(s)
    print("Number of unique filtered families:")
    print(len(s))
    values = range(0, len(s))
    return {k: v for k, v in zip(keys, values)}


# filtering out families with less than 100 samples:
family_counts = dataset['family'].value_counts()
families_to_keep = family_counts[family_counts>=100].index
# family_counts.to_csv('out.csv') 

dataset_filtered = dataset[dataset['family'].isin(families_to_keep)]
labels_filtered = dataset_filtered["family"]
unique_filtered_labels = set(labels_filtered.unique())
# dataset_filtered['family'].value_counts().to_csv('filtered_families.csv')

BODMAS_FILTERED_LABEL_MAPPINGS = set_to_dict(unique_filtered_labels)
print(BODMAS_FILTERED_LABEL_MAPPINGS)


ds_train, ds_valid = train_test_split(dataset_filtered, stratify=labels_filtered, test_size=0.25)

class BodmasDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform
        self.label_transform=lambda data: torch.tensor(data, dtype=torch.int),
        self.class2index = BODMAS_FILTERED_LABEL_MAPPINGS

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        line = self.df.iloc[index]
        filename = line["md5"]
        label = self.class2index[line["family"]]

        #TODO: read with opencv instead? - depends on the augm library
        # image = PIL.Image.open(os.path.join(data_dir, images_dir, filename + '_False_False.png'))
        image = cv2.imread(
            os.path.join(data_dir, images_dir, filename + '_224x224_True_True.png')
        ) 

        if self.transform is not None:
            # image = self.transform(image)
            image = self.transform(image=image)["image"]  # transformations with Albumentations
        
        label = torch.tensor(label)

        return image, label


# image transformations:
normalize_augment = A.Compose(
            [
                A.ToFloat(    # [0..256] --> [0..1]
                    max_value=256
                ),  
                A.Normalize(
                    mean=mean, std=std, max_pixel_value=1.0, p=1.0
                ),
                # remove cropping - I added it so that the images fit in GPU memory; shouldn't be necessary for images equal to or smaller in size than 300x300
                # A.crops.transforms.CenterCrop(
                #     256, 256, always_apply=True, p=1.0
                # ),
                ToTensorV2(),
            ]
)



# train set
train_dataset = BodmasDataset(  # for RoadType and RoadCondition
    df = ds_train,
    transform = normalize_augment
)


train_sampler = torch.utils.data.RandomSampler(
    train_dataset,
    replacement=False,
    num_samples=5000,
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
    transform = normalize_augment
)

valid_sampler = torch.utils.data.RandomSampler(
    valid_dataset,
    replacement=False,
    num_samples=500,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    # add to work with only a random subset of the original dataset
    # sampler=valid_sampler,
)
