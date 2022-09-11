import os

import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import wandb
import random

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, densenet121, DenseNet121_Weights
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn, optim
from sklearn.metrics import roc_auc_score

from attention_model_dominik.attention_model import prepare_attention_model
from attention_model_dominik.dataset import DRACEmbeddingDataset
from metric_classification import quadratic_weighted_kappa


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_binary(x):
    """
    Use to map labels column of dataframe with 3 classes to binary
    :param x: original label
    :return: 3 values for 0vs1, 0vs2 and 1vs2. In the case of no comparison it returns None
    """

    if x == 0:
        return 0, 0, None
    elif x == 1:
        return 1, None, 0
    elif x == 2:
        return None, 1, 1
    else:
        raise Exception("Only three classes supported!")


class DracClassificationDatasetTrain(Dataset):

    def __init__(self,
                 images_folder: str,
                 labels_csv: str,
                 transform: transforms.Compose,
                 task: str,
                 use_for_attention: bool = False):

        self.labels = pd.read_csv(labels_csv)
        self.img_path = images_folder
        self.transform = transform
        self.use_for_attention = use_for_attention

        if task == 'b':
            label = 'image quality level'
        elif task == 'c':
            label = 'DR grade'
        else:
            raise Exception("only b and c allowed!")

        # convert labels to binary
        self.labels['0vs1'], self.labels['0vs2'], self.labels['1vs2'] = \
            zip(*self.labels[label].map(to_binary))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_file = self.labels.iloc[idx, 0]
        label = self.labels.iloc[idx, 1]  # use non-binary classification
        label_bce = self.labels.iloc[idx, -3:]  # binary classification
        mask_bce = label_bce.notna()

        if not self.use_for_attention:
            image = np.array(Image.open(self.img_path + img_file))
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
            return self.transform(image), label, torch.tensor(label_bce), torch.tensor(mask_bce)

        else:
            image_patches = torch.load(self.img_path + img_file.split('.')[0] + '.pt')
            return image_patches, label



    @property
    def targets(self):
        return self.labels


class DracClassificationDatasetTest(Dataset):

    def __init__(self,
                 images_folder: str,
                 transform: transforms.Compose,
                 use_for_attention: bool = False):
        self.img_path = images_folder
        self.images = [file for file in os.listdir(images_folder)]
        self.transform = transform
        self.use_for_attention = use_for_attention

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_id = img_file

        if not self.use_for_attention:
            image = np.array(Image.open(self.img_path + img_file))
            image = np.repeat(image[..., np.newaxis], 3, axis=2)

            return self.transform(image), img_id
        else:
            image_patches = torch.load(self.img_path + img_file + '.pt')
            return image_patches, img_id


class DracClassificationModel(nn.Module):
    """ Variant of a pretrained network from torchvision
    for classifying images from the Drac2022 dataset.
    """

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
    """
    Initialises a model with default weights from torchvision models
    :param model: One of the following models are supported: ResNet50, ConvNeXt_tiny, DenseNet121 and EfficientNet_B0
    :param dropout: float between 0 and 1
    :return: torchvision model with default weights
    """

    if model == 'ResNet50':
        _model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model == 'ConvNeXt_tiny':
        _model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    elif model == 'EfficientNet_B0':
        _model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    elif model == 'DenseNet121':
        _model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    elif model == 'Attention - EfficientNet_B0':
        return prepare_attention_model(model, num_classes=3)
    elif model == 'Attention - ConvNeXt_tiny':
        return prepare_attention_model(model, num_classes=3)
    else:
        raise Exception("Only ResNet50, ConvNeXt_tiny, DenseNet121, EfficientNet_B0 "
                        "and 'Attention - EfficientNet_B0' allowed!")

    return DracClassificationModel(_model,
                                   flattened_size=1000,
                                   dropout=dropout)


def init_optimizer(params, lr, weight_decay, optimizer: str = 'Adam'):

    if optimizer == 'Adam':
        opt = optim.Adam(params=params,
                         lr=lr,
                         weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        opt = optim.AdamW(params=params,
                          lr=lr,
                          weight_decay=weight_decay)
    else:
        opt = None

    return opt


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> list:
    network.eval()
    errors = []
    y_array = []
    y_hat_array = []
    y_scores_array = []

    device = next(network.parameters()).device

    for batch_idx, (x, y, y_bce, mask) in enumerate(data):
        x, y, y_bce, mask = x.float().to(device), y.to(device), y_bce.float().to(device), mask.to(device)

        metric.set_mask(mask)
        y_bce = torch.nan_to_num(y_bce)

        y_hat = network(x)
        errors.append(metric(y_hat, y_bce).item())

        y_sigmoid = torch.sigmoid(y_hat)

        class_0 = (1 - y_sigmoid[:, 0]) + (1 - y_sigmoid[:, 1])
        class_1 = (y_sigmoid[:, 0]) + (1 - y_sigmoid[:, 2])
        class_2 = y_sigmoid[:, 1] + y_sigmoid[:, 2]

        y_hat_before_softmax = torch.vstack((class_0, class_1, class_2)).T

        y_scores = nn.functional.softmax(y_hat_before_softmax, 1).cpu().detach().numpy()

        y_array.append(y.cpu().detach().numpy())
        y_hat_array.append(torch.argmax(y_hat_before_softmax, dim=1).cpu().detach().numpy())
        y_scores_array.append(y_scores)

    kw_epoch = quadratic_weighted_kappa(np.hstack(y_array), np.hstack(y_hat_array))
    wandb.log({"validation/epoch quadratic weighted kappa": kw_epoch})

    auc_epoch = roc_auc_score(np.hstack(y_array), np.vstack(y_scores_array), average="macro", multi_class='ovo')
    wandb.log({"validation/epoch macro-AUC-ovo": auc_epoch})

    wandb.log({"validation/conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                                  y_true=np.hstack(y_array),
                                                                  preds=np.hstack(y_hat_array),
                                                                  class_names=["1", "2", "3"])})

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

    for batch_idx, (x, y, y_bce, mask) in enumerate(data):
        x, y, y_bce, mask = x.float().to(device), y.to(device), y_bce.float().to(device), mask.to(device)

        loss.set_mask(mask)  # set the mask for the current batch
        y_bce = torch.nan_to_num(y_bce)  # set NANs to 0 at the beginning otherwise it will mess up backprop

        y_hat = network(x)
        output = loss(y_hat, y_bce)
        output.backward()
        opt.step()
        opt.zero_grad()

        errors.append(output.item())

        y_sigmoid = torch.sigmoid(y_hat)

        class_0 = (1 - y_sigmoid[:, 0]) + (1 - y_sigmoid[:, 1])
        class_1 = (y_sigmoid[:, 0]) + (1 - y_sigmoid[:, 2])
        class_2 = y_sigmoid[:, 1] + y_sigmoid[:, 2]

        y_hat_before_softmax = torch.vstack((class_0, class_1, class_2)).T

        y_scores = nn.functional.softmax(y_hat_before_softmax, 1).cpu().detach().numpy()

        y_array.append(y.cpu().detach().numpy())
        y_hat_array.append(torch.argmax(y_hat_before_softmax, dim=1).cpu().detach().numpy())
        y_scores_array.append(y_scores)

    kw_epoch = quadratic_weighted_kappa(np.hstack(y_array), np.hstack(y_hat_array))
    wandb.log({"train/epoch quadratic weighted kappa": kw_epoch})

    auc_epoch = roc_auc_score(np.hstack(y_array), np.vstack(y_scores_array), average="macro", multi_class='ovo')
    wandb.log({"train/epoch macro-AUC-ovo": auc_epoch})

    wandb.log({"train/conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                             y_true=np.hstack(y_array),
                                                             preds=np.hstack(y_hat_array),
                                                             class_names=["1", "2", "3"])})

    return errors


def prepare_transform(base_path: str, image_folder: str, model: str = 'ResNet50'):

    if not os.path.exists(base_path + "mean.pt"):
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

    if model == 'ConvNeXt_tiny':
        transform = {
            "train": transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize([600, 600]),
                 transforms.RandomCrop(512),
                 transforms.RandomVerticalFlip(),
                 transforms.Normalize(mean=mean,
                                      std=std)]),
            "test": transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize([512, 512]),
                 transforms.Normalize(mean=mean,
                                      std=std)])
        }
    elif model == 'EfficientNet_B0':
        transform = {
            "train": transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize([620, 620]),
                 transforms.RandomCrop(600),
                 transforms.RandomVerticalFlip(),
                 transforms.Normalize(mean=mean,
                                      std=std)]),
            "test": transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize([600, 600]),
                 transforms.Normalize(mean=mean,
                                      std=std)])
        }
    else:
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
                                   model: str,
                                   num_workers: int = 4,
                                   task: str = 'b',
                                   seed: int = 7,
                                   use_for_attention: bool = False):

    g = torch.Generator()
    g.manual_seed(seed)

    transform = prepare_transform(base_path, image_folder, model)

    if model == "Attention - EfficientNet_B0":
        embedding_dir = "embeddings-efficientnet-train"

        data_train_valid = DRACEmbeddingDataset(
            annotations_file=labels_csv,
            emb_dir=os.path.join(base_path, embedding_dir)
        )
    elif model == "Attention - ConvNeXt_tiny":
        embedding_dir = "embeddings-ConvNeXt_tiny-glorious-sweep-3-train"

        data_train_valid = DRACEmbeddingDataset(
            annotations_file=labels_csv,
            emb_dir=os.path.join(base_path, embedding_dir)
        )
    else:
        data_train_valid = DracClassificationDatasetTrain(image_folder, labels_csv, transform["train"],
                                                          task, use_for_attention)

    if task == 'b':
        data_train, data_valid = torch.utils.data.random_split(data_train_valid, [600, 65], generator=g)
    elif task == 'c':
        data_train, data_valid = torch.utils.data.random_split(data_train_valid, [560, 51], generator=g)
    else:
        raise Exception("Only Tasks a or b allowed!")

    # Weighted sampling for train data
    targets = pd.read_csv(labels_csv)
    train_target = targets.iloc[data_train.indices, 1]
    train_class_sample_count = np.array(
        [len(np.where(train_target == t)[0]) for t in np.unique(train_target)])

    train_weight = 1. / train_class_sample_count

    train_samples_weight = np.array([train_weight[t] for t in train_target])

    train_samples_weight = torch.from_numpy(train_samples_weight).double()

    train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight))

    dataloader_train = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,
                                  worker_init_fn=seed_worker, generator=g)
    dataloader_valid = DataLoader(data_valid, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  worker_init_fn=seed_worker, generator=g)

    return dataloader_train, dataloader_valid, train_target


# Ideas from http://www.bioinf.jku.at/publications/2014/NIPS2014c.pdf
class MaskedBCE(nn.Module):

    def __init__(self):
        super(MaskedBCE, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.mask = None
        self.eps = 1e-5  # for numerical stability

    def forward(self, inputs, targets):

        if self.mask is None:
            raise Exception('Set mask before computing the loss.')

        part1 = targets * torch.log(self.sigmoid(inputs) + self.eps)
        part2 = (1 - targets) * torch.log(1 - self.sigmoid(inputs) + self.eps)
        part3 = torch.add(part1, part2)

        masked_bce = - torch.sum(self.mask * part3)  # mask out examples without experiment results

        return masked_bce

    def set_mask(self, mask):  # used to set the mask per batch
        self.mask = mask
