from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from Attention import Attention

from metric_classification import quadratic_weighted_kappa, roc_auc_score

torch.manual_seed(1806)
torch.cuda.manual_seed(1806)

@torch.enable_grad()
def update(
        network: nn.Module,
        data: DataLoader,
        loss: nn.Module,
        opt: optim.Optimizer
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Update the network to minimise some loss using a given optimiser.

    Parameters
    ----------
    network : nn.Module
        Pytorch module representing the network.
    data : DataLoader
        Pytorch dataloader that is able to
        efficiently sample mini-batches of data.
    loss : nn.Module
        Pytorch function that computes a scalar loss
        from the network logits and true data labels.
    opt : optim.Optimiser
        Pytorch optimiser to use for minimising the objective.

    Returns
    -------
    errors : tuple
        The computed loss, quadratic weighted kappa and auc on the provided data.
        The true labels as well as the predicted labels for the provided data.
    """
    network.train()
    device = next(network.parameters()).device

    tmp_ce, tmp_outputs, tmp_labels, tmp_scores = [], [], [], []
    for inputs, labels in data:
        opt.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        output, _, _ = network.forward(inputs)
        batch_loss = loss(output, labels)
        tmp_ce.append(batch_loss.item())
        batch_loss.backward()
        opt.step()
        tmp_ce.append(loss(output, labels).item())
        tmp_outputs.append(output.cpu().detach().numpy())
        tmp_labels.append(labels.cpu().detach().numpy())
        y_scores = nn.functional.softmax(output, dim=1).cpu().detach().numpy()
        tmp_scores.append(y_scores)

    y_true = np.concatenate(tmp_labels).ravel()
    y_hat = np.argmax(np.concatenate(tmp_outputs), axis=1)

    kappa = quadratic_weighted_kappa(
        y_true,
        y_hat)
    auc = roc_auc(
        y_true,
        np.concatenate(tmp_scores))

    metrics_log = {
        "train/ce-loss": torch.tensor(tmp_ce).mean().item(),
        "train/kappa": kappa,
        "train/auc": auc
    }

    return metrics_log, y_true, y_hat


@torch.no_grad()
def evaluate(
        network: nn.Module,
        data: DataLoader,
        loss: nn.Module
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate the performance of a network on some metric.

    Parameters
    ----------
    network : nn.Module
        Pytorch module representing the network.
    data : DataLoader
        Pytorch dataloader that is able to
        efficiently sample mini-batches of data.
    loss : callable
        Function that computes a scalar metric
        from the network logits and true data labels.
        The function should expect pytorch tensors as inputs.

    Returns
    -------
    metrics_log : tuple
        The computed loss, quadratic weighted kappa and auc on the provided data.
        The true labels as well as the predicted labels for the provided data.
    """
    network.eval()
    device = next(network.parameters()).device

    tmp_ce, tmp_outputs, tmp_labels, tmp_scores = [], [], [], []

    with torch.no_grad():
        for inputs, labels in data:
            inputs, labels = inputs.to(device), labels.to(device)
            output, _, _ = network(inputs)
            tmp_ce.append(loss(output, labels).item())
            tmp_outputs.append(output.cpu().numpy())
            tmp_labels.append(labels.cpu().numpy())
            y_scores = nn.functional.softmax(output, dim=1).cpu().numpy()
            tmp_scores.append(y_scores)

    y_true = np.concatenate(tmp_labels).ravel()
    y_hat = np.argmax(np.concatenate(tmp_outputs), axis=1)
    kappa = quadratic_weighted_kappa(
        y_true,
        y_hat)
    auc = roc_auc_score(
        y_true,
        np.concatenate(tmp_scores))
    metrics_log = {
        "validation/ce-loss": torch.tensor(tmp_ce).mean().item(),
        "validation/kappa": kappa,
        "validation/auc": auc
    }

    return metrics_log, y_true, y_hat


def prepare_attention_model(
        model_name: str,
        num_classes: int
) -> nn.Module:

    if model_name == "Attention - ResNet50":
        model = Attention(L=2048, num_classes=num_classes)

    elif model_name == "Attention - EfficientNet_v2":
        model = Attention(L=1280, num_classes=num_classes)

    elif model_name == "SwinTransformer":
        raise ValueError(f"Model: {model_name} not supported!")

    elif model_name == "Attention - ConvNext":
        model = Attention(L=1000, num_classes=num_classes)

    else:
        raise ValueError(f"Model: {model_name} not supported!")

    return model
