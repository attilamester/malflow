import os
from typing import Dict, Tuple

import albumentations as alb
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler

from core.processors.bagnet.dataset.dataset import Datasets, ImgDataset
from core.processors.bagnet.dataset.preprocess import filter_ds_having_at_column_min_occurencies
from util.logger import Logger


class BodmasDataset(Dataset):
    dataset: ImgDataset
    df: pd.DataFrame  # a subset of the ground-truth dataframe
    transform: alb.Compose
    family_index: Dict[str, int]

    def __init__(self, dataset: ImgDataset, df: pd.DataFrame, family_index: Dict[str, int],
                 transform: alb.Compose = None):
        self.dataset = dataset
        self.df = df
        self.family_index = family_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        line = self.df.iloc[index]
        label = self.family_index[line["family"]]
        filename = f"{line['md5']}_{self.dataset.img_shape[0]}x{self.dataset.img_shape[1]}_True_True.png"
        image = cv2.imread(os.path.join(self.dataset.img_dir_path, filename))

        if self.transform is not None:
            image = self.transform(image=image)["image"]  # transformations with Albumentations

        label = torch.tensor(label)

        return image, label


def get_transform_alb_norm(mean: float, std: float) -> alb.Compose:
    return alb.Compose([
        alb.ToFloat(max_value=256),  # [0..256] --> [0..1]
        alb.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
        # remove cropping according to device GPU ram
        # A.crops.transforms.CenterCrop(
        #     256, 256, always_apply=True, p=1.0
        # ),
        alb.pytorch.ToTensorV2(),
    ])


def get_train_valid_dataset_sampler_loader(dataset: ImgDataset) \
        -> Tuple[
            BodmasDataset, RandomSampler, DataLoader,
            BodmasDataset, RandomSampler, DataLoader]:
    Logger.info("Loading the dataset...")
    df = dataset.read_ground_truth()
    df_filtered, families_to_keep, family_index = filter_ds_having_at_column_min_occurencies(
        df, "family", 100)

    ds_train, ds_valid = train_test_split(df_filtered, stratify=families_to_keep, test_size=0.25)

    # train set
    train_dataset = BodmasDataset(dataset=dataset, df=ds_train,
                                  family_index=family_index,
                                  transform=get_transform_alb_norm(dataset.mean, dataset.std))
    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=Datasets.BODMAS.value.train_batch_size,
                              # when running with a subset of the dataset, set Shuffle to False, otherwise to True
                              shuffle=True,
                              # sampler=train_sampler,  # OPTIONAL; for running with a subset of the whole dataset
                              )

    # validation set
    valid_dataset = BodmasDataset(dataset=dataset, df=ds_valid,
                                  family_index=family_index,
                                  transform=get_transform_alb_norm(dataset.mean, dataset.std))
    valid_sampler = RandomSampler(valid_dataset, replacement=False, num_samples=500)
    valid_loader = DataLoader(valid_dataset, batch_size=Datasets.BODMAS.value.test_batch_size,
                              shuffle=True)  # sampler=valid_sampler,

    return train_dataset, train_sampler, train_loader, valid_dataset, valid_sampler, valid_loader
