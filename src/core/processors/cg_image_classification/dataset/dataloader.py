import os
from typing import Tuple, NamedTuple

import albumentations as alb
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from core.processors.cg_image_classification.dataset.dataset import ImgDataset
from util.logger import Logger


class BodmasDataset(Dataset):
    dataset: ImgDataset
    df: pd.DataFrame  # a subset of the ground-truth dataframe
    transform: alb.Compose

    def __init__(self, dataset: ImgDataset, df: pd.DataFrame, transform: alb.Compose = None):
        self.dataset = dataset
        self.df = df
        self.transform = transform
        self.iter_get_details = False

    def __len__(self):
        return len(self.df)

    def set_iter_details(self, flag: bool):
        """
        Setting True should be used only for debugging purposes - on train loop, this will raise an exception
        """
        self.iter_get_details = flag

    class ItemDetailsPacked(NamedTuple):
        packed: bool
        orig_md5: str = ""
        orig_filename: str = ""
        orig_image: torch.Tensor = ""

    class ItemDetails(NamedTuple):
        md5: str
        filename: str
        packed: "BodmasDataset.ItemDetailsPacked"

    def __getitem__(self, index):
        line = self.df.iloc[index]
        label = self.dataset.data_class2index[line["family"]]
        label = torch.tensor(label)

        def get_filename(md5):
            return f"{md5}_{self.dataset.img_shape[0]}x{self.dataset.img_shape[1]}_True_True.png"

        def read_image(filename):
            image = cv2.imread(os.path.join(self.dataset.img_dir_path, filename))
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image

        if pd.notna(line["unpacked-md5"]):
            md5 = line["unpacked-md5"]
        else:
            md5 = line["md5"]

        filename = get_filename(md5)
        image = read_image(filename)

        if not self.iter_get_details:
            """
            Normal flow | during training / evaluating
            """
            return image, label

        """
        Debug flow | when we want to get more details about the sample
        """

        if pd.notna(line["unpacked-md5"]):
            # we already have the unpacked-md5 + image above
            filename_original = get_filename(line["md5"])
            image_original = read_image(filename_original)
            details = BodmasDataset.ItemDetails(md5, filename,
                                                BodmasDataset.ItemDetailsPacked(True, line["md5"], filename_original,
                                                                                image_original))
        else:
            details = BodmasDataset.ItemDetails(md5, filename, BodmasDataset.ItemDetailsPacked(False))

        return image, label, details


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


def create_torch_bodmas_dataset_loader(dataset: ImgDataset, subset_df: pd.DataFrame, batch_size: int) -> Tuple[
    BodmasDataset, DataLoader]:
    ds = BodmasDataset(dataset=dataset, df=subset_df, transform=get_transform_alb_norm(dataset.mean, dataset.std))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return ds, dl


def create_bodmas_train_val_loader(dataset: ImgDataset, items_per_class: int, batch_size: int) \
        -> Tuple[
            BodmasDataset, DataLoader,
            BodmasDataset, DataLoader]:
    Logger.info(f"Creating dataset & loader with items_per_class:{items_per_class}, batch_size:{batch_size}")

    df_filtered = dataset.filter_ground_truth(items_per_class)

    ds_train, ds_valid = train_test_split(df_filtered, stratify=df_filtered["family"], test_size=0.25)

    ds_tr, dl_tr = create_torch_bodmas_dataset_loader(dataset, ds_train, batch_size)
    ds_va, dl_va = create_torch_bodmas_dataset_loader(dataset, ds_valid, batch_size)

    return ds_tr, dl_tr, ds_va, dl_va
