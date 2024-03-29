import pandas as pd
import torch
import wandb

from torch import nn

from helpers import init_model, update, evaluate, prepare_classification_dataset, init_optimizer, MaskedBCE
from helpers_attention import update_attention, evaluate_attention

if __name__ == "__main__":

    TASK = 'c'  # for now only b and c work (classification)
    # "Classification B Quality" "Classification C Grading"
    TASK_DESC = "Classification C Grading"  # for logging only
    DATA_ROOT = "/system/user/publicdata/dracch/"  # "/Users/markus/Downloads/DRAC2022/"

    # Prepare paths
    if TASK == 'a':
        base_path = DATA_ROOT + "A. Segmentation"
        x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
        y_train_raw_path = base_path + "/2. Groundtruths/a. Training Set/"
    elif TASK == 'b':
        base_path = DATA_ROOT + "B. Image Quality Assessment"
        x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
        y_train_raw_path = base_path + "/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv"
    elif TASK == 'c':
        base_path = DATA_ROOT + "C. Diabetic Retinopathy Grading"
        x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
        y_train_raw_path = base_path + "/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv"
    else:
        raise Exception("Only Tasks a,b or c allowed!")

    sweep_config = {
        "method": "random",
        "metric": {
            "name": "validation/epoch quadratic weighted kappa",
            "goal": "maximize"
        },
        "parameters": {
            "model": {
                # "ResNet50", "EfficientNet_B0", "ConvNeXt_tiny", Attention - EfficientNet_B0, Attention - ConvNeXt_tiny
                "values": ["Attention - ConvNeXt_tiny"]
            },
            "task": {
                "values": [TASK_DESC]
            },
            "epochs": {
                "values": [10, 20, 30, 40]
            },
            "learning_rate": {
                "min": 0.0000001,
                "max": 0.0001
            },
            "weight_decay": {
                "values": [0., 0.0005, 0.005, 0.05]
            },
            "batch_size": {
                "values": [16, 64]
            },
            "dropout": {
                "values": [0., 0.3, 0.6, 0.8]
            },
            "optimizer": {
                "values": ["Adam", "AdamW"]
            },
            "use_weighted_ce": {
                "values": [True]
            },
            "use_masked_bce": {
                "values": [False]
            },
            "seed": {
                "values": [7, 934, 314]
            }
        }
    }


    def train():
        with wandb.init() as run:
            config = wandb.config

            torch.manual_seed(config["seed"])

            model = init_model(config.model, config.dropout)
            device = torch.device("cuda:0")
            model.to(device)

            if config["task"] == "Classification B Quality":
                task = 'b'
            elif config["task"] == "Classification C Grading":
                task = 'c'
            else:
                raise Exception("Wrong Task Description!")

            dataloader_train, dataloader_valid, train_target = prepare_classification_dataset(base_path,
                                                                                              x_train_raw_path,
                                                                                              y_train_raw_path,
                                                                                              config["batch_size"],
                                                                                              config["model"],
                                                                                              num_workers=8,
                                                                                              seed=config["seed"],
                                                                                              task=task
                                                                                              )

            opt = init_optimizer(model.parameters(),
                                 config["learning_rate"],
                                 config["weight_decay"],
                                 config["optimizer"])

            if config["use_weighted_ce"]:
                ce_weight = torch.tensor(max(train_target.value_counts()) / train_target.value_counts()).flip(0).float()
                ce_weight = ce_weight.to(device)
            else:
                ce_weight = None

            ce = nn.CrossEntropyLoss(weight=ce_weight)

            if config["use_masked_bce"]:
                ce = MaskedBCE()

            for epoch in range(config["epochs"]):
                if config["model"] == 'Attention - EfficientNet_B0' or config["model"] == 'Attention - ConvNeXt_tiny':
                    local_errs = update_attention(model, dataloader_train, ce, opt)
                    local_val_errs = evaluate_attention(model, dataloader_valid, ce)
                else:
                    local_errs = update(model, dataloader_train, ce, opt)
                    local_val_errs = evaluate(model, dataloader_valid, ce)

                wandb.log({"train/error": sum(local_errs) / len(local_errs),
                           "validation/error": sum(local_val_errs) / len(local_val_errs)})

                if (epoch + 1) % 10 == 0:
                    model_base_path = "/system/user/publicwork/student/karner/model-weights/"
                    path = model_base_path + f"model_{run.name}_{epoch + 1}.pth"

                    torch.save(model.state_dict(), path)

                    artifact = wandb.Artifact(f'model_{run.name}', type='model')
                    artifact.add_file(path)
                    run.log_artifact(artifact)

    continue_sweep_id = None

    if not continue_sweep_id:
        sweep_id = wandb.sweep(sweep_config, entity="markuskarner", project="DRAC2022")

        count = 50  # number of runs to execute
        wandb.agent(sweep_id, function=train, count=count)
    else:
        wandb.agent(continue_sweep_id, function=train, project="DRAC2022")
