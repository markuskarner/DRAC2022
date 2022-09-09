import os
from glob import glob
from typing import Tuple, List, Dict, Optional, Any

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


class DRACImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.GRAY) / 255
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


class DRACPatchDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path, mode=ImageReadMode.GRAY) / 255
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label, img_name


class DRACEmbeddingDataset(Dataset):
    def __init__(self, annotations_file, emb_dir):
        self.emb_labels = pd.read_csv(annotations_file)
        self.emb_dir = emb_dir

    def __len__(self):
        return len(self.emb_labels)

    def __getitem__(self, idx):
        emb_name = self.emb_labels.iloc[idx, 0]
        emb_name = emb_name.split(".")[0] + ".pt"
        emb_path = os.path.join(self.emb_dir, emb_name)
        emb = torch.load(emb_path)
        label = self.emb_labels.iloc[idx, 1]

        return emb, label


class DRACEmbeddingPredDataset(Dataset):
    def __init__(self, emb_dir):
        self.embs = sorted(glob(os.path.join(emb_dir, "*.pt")))
        self.emb_dir = emb_dir

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        emb_path = self.embs[idx]
        emb_name = os.path.basename(emb_path).replace(".pt", ".png")
        emb = torch.load(emb_path)

        return emb, emb_name


class DRACPredictionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.images = sorted(glob(os.path.join(img_dir, "*.png")))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)
        image = read_image(img_path, mode=ImageReadMode.GRAY) / 255
        if self.transform:
            image = self.transform(image)
        return image, img_name


def channel_copy(data: torch.tensor):
    return data.repeat(3, 1, 1)


def prepare_datasets(
        model_name: str,
        mean: List[float],
        std: List[float],
        annotation_files: Dict[str, str],
        image_directories: Dict[str, str],
        dataloader_params: Dict[str, Any],
        sample_weights: Optional[np.array] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the data sets for training and validation

    Args:
        mean: Image mean
        std: Image standard deviation
        annotation_files: File paths for annotation files of the 'train' and 'val' data set
        image_directories: Directories containing the images of the 'train' and 'val' data set
        dataloader_params: Parameters for the dataloader
        sample_weights: Weights of all the samples in the train split. If given a weighted sampler is
            used for the train data loader

    Returns:
        Dataloaders for the train and validation splits
    """
    if model_name == "ResNet50":
        resize, crop = 256, 224
    elif model_name == "EfficientNet_v2":
        resize, crop = 400, 386
    elif model_name == "SwinTransformer":
        resize, crop = 580, 512
    else:
        raise ValueError(f"Model: {model_name} not supported!")

    image_transform = {
        "train": transforms.Compose([
            channel_copy,
            transforms.Resize((resize, resize)),
            transforms.RandomCrop((crop, crop)),
            transforms.Normalize(
                mean=mean,
                std=std
            ),
        ]),
        "val": transforms.Compose([
            channel_copy,
            transforms.Resize((crop, crop)),
            transforms.Normalize(
                mean=mean,
                std=std
            ),
        ])
    }

    train_set = DRACImageDataset(
        annotations_file=annotation_files["train"],
        img_dir=image_directories["train"],
        transform=image_transform['train']
    )

    val_set = DRACImageDataset(
        annotations_file=annotation_files["val"],
        img_dir=image_directories["val"],
        transform=image_transform['val']
    )

    train_sampler = None

    if sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor([sample_weights[i] for i in train_set.img_labels["image quality level"]]),
            num_samples=len(train_set),
            replacement=False
        )

    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        **dataloader_params
    )

    val_loader = DataLoader(
        val_set,
        **dataloader_params
    )

    return train_loader, val_loader
