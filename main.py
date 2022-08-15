import pandas as pd
import torch
import wandb

from torch import nn
from torch import optim


from helpers import init_model, update, evaluate, prepare_classification_dataset

if __name__ == "__main__":

    TASK = 'b'  # for now only b and c work (classification)
    TASK_DESC = "Classification B Quality"  # for logging only

    # Prepare paths
    if TASK == 'a':
        base_path = "/Users/markus/Downloads/DRAC2022/A. Segmentation/"
        x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
        y_train_raw_path = base_path + "/2. Groundtruths/a. Training Set/"
    elif TASK == 'b':
        base_path = "/Users/markus/Downloads/DRAC2022/B. Image Quality Assessment/"
        x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
        y_train_raw_path = base_path + "/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv"
    elif TASK == 'c':
        base_path = "/Users/markus/Downloads/DRAC2022/C. Diabetic Retinopathy Grading/"
        x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
        y_train_raw_path = base_path + "/2. Groundtruths/" \
                                         "a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv"
    else:
        raise Exception("Only Tasks a,b or c allowed!")

    labels = pd.read_csv(y_train_raw_path)

    sweep_config = {
        "method": "random",
        "metric": {
            "name": "quadratic weighted kappa",
            "goal": "maximize"
        },
        "parameters": {
            "model": {
                "values": ["ConvNeXt_tiny"]
            },
            "task": {
                "values": [TASK_DESC]
            },
            "epochs": {
                "values": [20, 30]
            },
            "learning_rate": {
                "min": 0.00001,
                "max": 0.1
            },
            "weight_decay": {
                "values": [0.0005, 0.005, 0.05]
            },
            "batch_size": {
                "values": [16, 32, 64, 128]
            },
            "dropout": {
                "values": [0.]  # [0., 0.3, 0.4, 0.5]
            },
        }
    }

    def train():
        with wandb.init():
            config = wandb.config

            model = init_model(config.dropout)
            device = torch.device("mps")
            model.to(device)

            dataloader_train, dataloader_valid = prepare_classification_dataset(base_path,
                                                                                x_train_raw_path,
                                                                                y_train_raw_path,
                                                                                config["batch_size"])

            opt = optim.Adam(model.parameters(),
                             lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])
            ce = nn.CrossEntropyLoss()

            for epoch in range(config["epochs"]):
                local_errs = update(model, dataloader_train, ce, opt)
                local_val_errs = evaluate(model, dataloader_valid, ce)

                wandb.log({"train error": sum(local_errs) / len(local_errs),
                           "validation error": sum(local_val_errs) / len(local_val_errs)})


    continue_sweep_id = None

    if not continue_sweep_id:
        sweep_id = wandb.sweep(sweep_config, entity="markuskarner", project="DRAC2022")

        count = 10  # number of runs to execute
        wandb.agent(sweep_id, function=train, count=count)
    else:
        wandb.agent(continue_sweep_id, function=train, project="DRAC2022")
