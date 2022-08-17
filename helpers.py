import os

import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import wandb
import random

from PIL import Image
from torchvision import models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn, optim
from sklearn.metrics import roc_auc_score

from DRAC2022_zhuanjiao.evaluation.metric_classification import confusion_matrix, histogram, quadratic_weighted_kappa


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DracClassificationDatasetTrain(Dataset):

    def __init__(self,
                 images_folder: str,
                 labels_csv: str,
                 transform: transforms.Compose):
        self.labels = pd.read_csv(labels_csv)
        self.img_path = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_file = self.labels.iloc[idx, 0]
        label = self.labels.iloc[idx, 1]

        image = np.array(Image.open(self.img_path + img_file))
        image = np.repeat(image[..., np.newaxis], 3, axis=2)

        return self.transform(image), label

    @property
    def targets(self):
        return self.labels


class DracClassificationDatasetTest(Dataset):

    def __init__(self,
                 images_folder: str,
                 transform: transforms.Compose):
        self.img_path = images_folder
        self.images = [file for file in os.listdir(images_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_id = int(img_file.split('.')[0])

        image = np.array(Image.open(self.img_path + img_file))
        image = np.repeat(image[..., np.newaxis], 3, axis=2)

        return self.transform(image), img_id


class DracClassificationModel(nn.Module):
    """ Variant of a pretrained network from torchvision
    for classifying images from the Drac2022 dataset.
    """

    # TODO down sample input images

    def __init__(self, model: nn.Module,
                 num_classes: int = 3,
                 flattened_size: int = 1000,
                 dropout: float = 0.):
        """
        Parameters
        ----------
        model : nn.Module
            A pretrained network.
        num_classes : int
            The number of output classes in the data.
        flattened_size: int
            Size of the flattened layer after AdaptiveAvgPool2d
        """
        super(DracClassificationModel, self).__init__()

        # self.downsampler = nn.Sequential(
        #     nn.Conv2d()
        # )
        self.model = model
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=flattened_size, out_features=num_classes)
        )

        # init classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


def init_model(model: str, dropout: float = 0.):
    # TODO try bigger convNeXt models as well
    if model == 'ResNet50':
        _model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model == 'ConvNeXt_tiny':
        _model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    else:
        raise Exception("Only ResNet50 and ConvNeXt_tiny allowed!")

    return DracClassificationModel(_model,
                                   flattened_size=1000,
                                   dropout=dropout)


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> list:
    network.eval()
    errors = []
    y_array = []
    y_hat_array = []
    y_scores_array = []

    device = next(network.parameters()).device

    for batch_idx, (x, y) in enumerate(data):
        x, y = x.float().to(device), y.to(device)

        y_hat = network(x)
        errors.append(metric(y_hat, y).item())

        # acc = balanced_accuracy_score(y.cpu(), torch.argmax(y_hat, dim=1).cpu().detach().numpy())
        # wandb.log({"validation accuracy": acc})

        y_scores = nn.functional.softmax(y_hat, 1).cpu().detach().numpy()

        y_array.append(y.cpu().detach().numpy())
        y_hat_array.append(torch.argmax(y_hat, dim=1).cpu().detach().numpy())
        y_scores_array.append(y_scores)

        #kw = quadratic_weighted_kappa(y.cpu().detach().numpy(), torch.argmax(y_hat, dim=1).cpu().detach().numpy())
        #wandb.log({"validation/batch quadratic weighted kappa": kw})

        #auc = roc_auc_score(y.cpu().detach().numpy(), y_scores, average="macro", multi_class='ovo')
        #wandb.log({"validation/batch macro-AUC-ovo": auc})

    kw_epoch = quadratic_weighted_kappa(np.hstack(y_array), np.hstack(y_hat_array))
    wandb.log({"validation/epoch quadratic weighted kappa": kw_epoch})

    auc_epoch = roc_auc_score(np.hstack(y_array), np.vstack(y_scores_array), average="macro", multi_class='ovo')
    wandb.log({"validation/epoch macro-AUC-ovo": auc_epoch})

    return errors


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: optim.Optimizer) -> list:
    network.train()
    errors = []
    y_array = []
    y_hat_array = []
    y_scores_array = []

    device = next(network.parameters()).device

    for batch_idx, (x, y) in enumerate(data):
        x, y = x.float().to(device), y.to(device)

        y_hat = network(x)
        output = loss(y_hat, y)
        output.backward()
        opt.step()
        opt.zero_grad()

        errors.append(output.item())

        # balanced_acc = balanced_accuracy_score(y.cpu(), torch.argmax(y_hat, dim=1).cpu().detach().numpy())
        # wandb.log({"train/balanced accuracy": balanced_acc})
        y_scores = nn.functional.softmax(y_hat, 1).cpu().detach().numpy()

        y_array.append(y.cpu().detach().numpy())
        y_hat_array.append(torch.argmax(y_hat, dim=1).cpu().detach().numpy())
        y_scores_array.append(y_scores)

        #kw = quadratic_weighted_kappa(y.cpu().detach().numpy(), torch.argmax(y_hat, dim=1).cpu().detach().numpy())
        #wandb.log({"train/batch quadratic weighted kappa": kw})

        #auc = roc_auc_score(y.cpu().detach().numpy(), y_scores, average="macro", multi_class='ovo')
        #wandb.log({"train/batch macro-AUC-ovo": auc})

    kw_epoch = quadratic_weighted_kappa(np.hstack(y_array), np.hstack(y_hat_array))
    wandb.log({"train/epoch quadratic weighted kappa": kw_epoch})
    #print("before epoch auc")
    #print(np.hstack(y_array).shape)
    #print(np.vstack(y_scores_array).shape)
    auc_epoch = roc_auc_score(np.hstack(y_array), np.vstack(y_scores_array), average="macro", multi_class='ovo')
    wandb.log({"train/epoch macro-AUC-ovo": auc_epoch})
    #print("after epoch auc")

    return errors


def prepare_transform(base_path: str, image_folder: str, calculate_mean_and_std: bool = False):

    if calculate_mean_and_std:
        images = image_folder  # b_x_train_raw_path
        ids = set()
        for file in os.listdir(images):
            ids.add(file)

        mean = 0
        std = 0
        for i in ids:
            img_tensor = transforms.ToTensor()(Image.open(images + i))

            mean += 1 / len(ids) * torch.mean(img_tensor, (1, 2))
            std += 1 / len(ids) * torch.std(img_tensor, (1, 2))

            torch.save(mean, base_path + "mean.pt")
            torch.save(std, base_path + "std.pt")
    else:
        mean = torch.load(base_path + "mean.pt")
        std = torch.load(base_path + "std.pt")

    print(f'mean:{mean}, std:{std}')

    transform = {
        "train": transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([256, 256]),
             transforms.RandomCrop(224),
             transforms.RandomVerticalFlip(),
             transforms.Normalize(mean=mean,
                                  std=std)]),
        "test": transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([224, 224]),
             transforms.Normalize(mean=mean,
                                  std=std)])
    }

    return transform


def prepare_classification_dataset(base_path: str,
                                   image_folder: str,
                                   labels_csv: str,
                                   batch_size: int,
                                   num_workers: int = 4):

    g = torch.Generator()
    g.manual_seed(7)

    transform = prepare_transform(base_path, image_folder, False)

    data_train_valid = DracClassificationDatasetTrain(image_folder, labels_csv, transform["train"])
    data_train, data_valid = torch.utils.data.random_split(data_train_valid, [600, 65], generator=g)

    # Weighted sampling for train data
    targets = pd.read_csv(labels_csv)
    train_target = targets.iloc[data_train.indices, 1]
    train_class_sample_count = np.array(
        [len(np.where(train_target == t)[0]) for t in np.unique(train_target)])
    #print(train_class_sample_count)
    train_weight = 1. / train_class_sample_count

    train_samples_weight = np.array([train_weight[t] for t in train_target])

    train_samples_weight = torch.from_numpy(train_samples_weight).double()

    train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight))

    # dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_train = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,
                                  worker_init_fn=seed_worker, generator=g)
    dataloader_valid = DataLoader(data_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  worker_init_fn=seed_worker, generator=g)

    return dataloader_train, dataloader_valid
