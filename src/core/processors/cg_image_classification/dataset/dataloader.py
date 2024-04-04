from typing import Tuple, NamedTuple, Dict

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
        self.iter_details = False

    def __len__(self):
        return len(self.df)

    def set_iter_details(self, flag: bool):
        """
        Setting True should be used only for debugging purposes - on train loop, this will raise an exception
        """
        self.iter_details = flag

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
        label = self.dataset._data_class2index[line["family"]]
        label = torch.tensor(label)

        def read_image(filename):
            image = self.dataset.read_image(filename)
            if self.transform is not None:
                image = self.transform(image=image)["image"]
            return image

        md5 = self.dataset.get_row_id(line)
        filename = self.dataset.get_filename(md5)
        image = read_image(filename)

        if not self.iter_details:
            """
            Normal flow | during training / evaluating
            """
            return image, label

        """
        Debug flow | when we want to get more details about the sample
        """

        if pd.notna(line["unpacked-md5"]):
            # we already have the unpacked-md5 + image above
            filename_original = self.dataset.get_filename(line["md5"])
            image_original = read_image(filename_original)
            details = BodmasDataset.ItemDetails(md5, filename,
                                                BodmasDataset.ItemDetailsPacked(True, line["md5"], filename_original,
                                                                                image_original))
        else:
            details = BodmasDataset.ItemDetails(md5, filename, BodmasDataset.ItemDetailsPacked(False))

        return image, label, details


def get_transform(mean: float, std: float, min_size: Tuple[int, int] = None) -> alb.Compose:
    if not min_size:
        min_size = (1, 1)

    return alb.Compose([
        # alb.ToFloat(max_value=256),  # [0..256] --> [0..1]
        alb.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
        alb.PadIfNeeded(min_height=min_size[0], min_width=min_size[1], position=alb.PadIfNeeded.PositionType.TOP_LEFT,
                        border_mode=cv2.BORDER_REFLECT_101),
        alb.pytorch.ToTensorV2(),
    ])


def create_torch_bodmas_dataset_loader(dataset: ImgDataset, subset_df: pd.DataFrame, batch_size: int,
                                       model_requirements: Dict = None) -> Tuple[
    BodmasDataset, DataLoader]:
    if not model_requirements:
        model_requirements = {}
    ds = BodmasDataset(dataset=dataset, df=subset_df,
                       transform=get_transform(dataset.mean, dataset.std, model_requirements.get("min_size", None)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return ds, dl


def create_bodmas_train_val_loader(dataset: ImgDataset, batch_size: int, model_requirements: Dict = None) \
        -> Tuple[
            BodmasDataset, DataLoader,
            BodmasDataset, DataLoader]:
    Logger.info(f"[Dataloader] Creating dataset loader with batch:{batch_size}")

    ds_train, ds_valid = train_test_split(dataset._data_df_gt_filtered_noaugm,
                                          stratify=dataset._data_df_gt_filtered_noaugm["family"],
                                          test_size=0.25)

    ds_train = pd.concat([ds_train, dataset._data_df_gt_filtered_augm])

    ds_tr, dl_tr = create_torch_bodmas_dataset_loader(dataset, ds_train, batch_size, model_requirements)
    ds_va, dl_va = create_torch_bodmas_dataset_loader(dataset, ds_valid, batch_size, model_requirements)

    return ds_tr, dl_tr, ds_va, dl_va
