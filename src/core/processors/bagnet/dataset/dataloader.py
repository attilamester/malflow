import os

import albumentations as alb
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler

from core.processors.bagnet.config import Datasets, ImgDataset


class BodmasDataset(Dataset):
    def __init__(self, df, transform: alb.Compose = None):
        self.df = df
        self.transform = transform
        self.label_transform = lambda data: torch.tensor(data, dtype=torch.int),
        #self.class2index = BODMAS_FILTERED_LABEL_MAPPINGS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        line = self.df.iloc[index]
        filename = line["md5"]
        label = self.class2index[line["family"]]

        # TODO: read with opencv instead? - depends on the augm library
        # image = PIL.Image.open(os.path.join(data_dir, images_dir, filename + '_False_False.png'))
        image = cv2.imread(
            os.path.join(data_dir, images_dir, filename + '_224x224_True_True.png')
        )

        if self.transform is not None:
            # image = self.transform(image)
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


def get_train_valid_dataset_sampler_loader(dataset: ImgDataset):
    # load the data
    # data_dir = dataset_config["data_dir"]
    # metadata_file = dataset_config["metadata_file"]
    # images_dir = dataset_config["images_dir"]
    # img_shape = dataset_config["img_shape"]
    # mean = dataset_config["mean"]
    # std = dataset_config["std"]
    dataset = pd.read_csv(os.path.join(data_dir, metadata_file), delimiter=",")

    def set_to_dict(s):
        keys = list(s)
        print("Number of unique filtered families:")
        print(len(s))
        values = range(0, len(s))
        return {k: v for k, v in zip(keys, values)}

    # filtering out families with less than 100 samples:
    family_counts = dataset['family'].value_counts()
    families_to_keep = family_counts[family_counts >= 100].index
    # family_counts.to_csv('out.csv')

    dataset_filtered = dataset[dataset['family'].isin(families_to_keep)]
    labels_filtered = dataset_filtered["family"]
    unique_filtered_labels = set(labels_filtered.unique())
    # dataset_filtered['family'].value_counts().to_csv('filtered_families.csv')

    BODMAS_FILTERED_LABEL_MAPPINGS = set_to_dict(unique_filtered_labels)
    print(BODMAS_FILTERED_LABEL_MAPPINGS)

    ds_train, ds_valid = train_test_split(dataset_filtered, stratify=labels_filtered, test_size=0.25)

    # train set
    train_dataset = BodmasDataset(df=ds_train, transform=get_transform_alb_norm(mean, std))

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=5000)

    train_loader = DataLoader(train_dataset, batch_size=Datasets.BODMAS.value.train_batch_size,
                              # when running with a subset of the dataset, set Shuffle to False, otherwise to True
                              shuffle=True,
                              # sampler=train_sampler,  # OPTIONAL; for running with a subset of the whole dataset
                              )

    # validation set
    valid_dataset = BodmasDataset(df=ds_valid, transform=get_transform_alb_norm(mean, std))
    valid_sampler = RandomSampler(valid_dataset, replacement=False, num_samples=500)
    valid_loader = DataLoader(valid_dataset, batch_size=Datasets.BODMAS.value.test_batch_size,
                              shuffle=True)  # sampler=valid_sampler,
