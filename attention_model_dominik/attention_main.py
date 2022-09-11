import os
from time import time, gmtime, strftime
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import prepare_datasets, DRACEmbeddingDataset
from metric_classification import quadratic_weighted_kappa, roc_auc_score

from Attention import Attention
from attention_model import prepare_attention_model, evaluate, update

torch.manual_seed(7)
torch.cuda.manual_seed(7)

import wandb


"""
Metrics:

quadratic-weighted Kappa
macro-AUC (Area Under Curve)
macro-precision
macro-sensitivity
macro-specificity

Note: "macro" means calculate metrics for each label, and find their unweighted mean.
"""


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
        output, hat, A = network.forward(inputs)
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
    auc = roc_auc_score(
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
            output, hat, A = network(inputs)
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


def train():
    start = time()
    dataloader_params = {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True
    }

    task = 'c'  # for now only b and c work (classification)
    # "Classification B Quality" "Classification C Grading"
    task_desc = "Classification C Grading"  # for logging only
    data_root = "/system/user/publicdata/dracch/"  # "/Users/markus/Downloads/DRAC2022/"


    if task == 'b':
        base_path = data_root + "B. Image Quality Assessment"
        x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
        y_train_raw_path = base_path + "/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv"
        class_names = ["Poor quality level (0)", "Good quality level (1)", "Excellent quality level (2)"]
    elif task == 'c':
        base_path = data_root + "C. Diabetic Retinopathy Grading"
        x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
        y_train_raw_path = base_path + "/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv"
        class_names = ["Normal (0)", "NPDR(1)", "PDR (2)"]
    else:
        raise Exception('Only Task a and b allowed.')

    run = wandb.init(entity="markuskarner", project="DRAC2022")
    config = wandb.config
    run.tags += (config["model"])

    if config["model_name"] == "Attention - EfficientNet_B0":
        embedding_dir = "embeddings-efficientnet-train"
    else:
        raise Exception('No valid model_name.')

    train_set = DRACEmbeddingDataset(
        annotations_file=os.path.join(y_train_raw_path),
        emb_dir=os.path.join(base_path, embedding_dir)
    )

    val_set = DRACEmbeddingDataset(
        annotations_file=os.path.join(quality_dir, "validation.csv"),
        emb_dir=os.path.join(quality_dir, "1. Original Images", embedding_dir)
    )

    train_loader = DataLoader(
        train_set,
        **dataloader_params
    )

    val_loader = DataLoader(
        val_set,
        **dataloader_params
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device for training: ", device)

    network = prepare_attention_model(config["model_name"], num_classes=3)
    # network.load_state_dict(
    #     torch.load(os.path.join(ckpt_dir, "lilac-sweep-8.pt"))
    # )
    network.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(
    #     network.parameters(),
    #     lr=config["lr"],
    #     weight_decay=config["weight_decay"]
    # )

    optimizer = optim.AdamW(
        network.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    val_metrics, val_true, val_pred = evaluate(
        network=network,
        data=val_loader,
        loss=criterion,
    )

    wandb.log(val_metrics)
    wandb.log({"validation/conf_mat": wandb.plot.confusion_matrix(
        probs=None,
        y_true=val_true,
        preds=val_pred,
        class_names=class_names
    )})

    og_kappa = val_metrics["validation/kappa"]

    for i in trange(config["epochs"]):
        train_metrics, train_true, train_pred = update(
            network=network,
            data=train_loader,
            loss=criterion,
            opt=optimizer,
        )

        wandb.log(train_metrics)
        wandb.log({"train/conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=train_true,
            preds=train_pred,
            class_names=class_names
        )})

        val_metrics, val_true, val_pred = evaluate(
            network=network,
            data=val_loader,
            loss=criterion,
        )

        wandb.log(val_metrics)
        wandb.log({"validation/conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=val_true,
            preds=val_pred,
            class_names=class_names
        )})

        if val_metrics["validation/kappa"] > og_kappa:
            torch.save(
                network.state_dict(),
                os.path.join(ckpt_dir, f"{run.name}.pt")
            )

            og_kappa = val_metrics["validation/kappa"]
            print(f"State Dict saved after epoch {i+1}")

        if (i+1) % 5 == 0:
            torch.save(
                network.state_dict(),
                os.path.join(ckpt_dir, f"{run.name}-epoch-{i+1}.pt")
            )

    print(f"Whole Training took: {strftime('%H:%M:%S', gmtime(time() - start))}")


if __name__ == '__main__':
    sweep_config = {
        "method": "random",
        "metric": {
            "name": "validation/kappa",
            "goal": "maximize"
        },
        "parameters": {
            "model": {
                "values": ["Attention - EfficientNet_B0"]
            },
            "task": {
                "values": ["Classification C Grading"]
            },
            "epochs": {
                "values": [15, 25, 30]
            },
            "lr": {
                "min": 0.000001,
                "max": 0.0005
            },
            "weight_decay": {
                "values": [0.0005, 0.005, 0.05]
            },
            "batch_size": {
                "values": [16,32,64,128]  # , 32, 64]
            },
            "use_weighted_sampling": {
                "values": [True, False]
            },
            "dropout": {
                "values": [0., 0.7, 0.8, 0.9]
            },
        }
    }

    continue_sweep_id = None

    if not continue_sweep_id:
        sweep_id = wandb.sweep(sweep_config, entity="elba", project="DRAC2022")

        count = 20  # number of runs to execute
        wandb.agent(sweep_id, function=train, count=count)
    else:
        wandb.agent(continue_sweep_id, function=train, project="DRAC2022")

    train()
