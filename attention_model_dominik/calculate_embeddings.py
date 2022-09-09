import wandb
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from helpers import init_model
from dataset import DRACPatchDataset


torch.manual_seed(7)
torch.cuda.manual_seed(7)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

@torch.no_grad()
def calc_embeddings(network: nn.Module, data: DataLoader):
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
    metrics_log : dict
        The computed loss, quadratic weighted kappa and auc on the provided data.
    """
    network.eval()
    _device = next(network.parameters()).device

    tmp_outputs, tmp_labels, tmp_img_names = [], [], []

    embeddings = {}

    with torch.no_grad():
        for i, (inputs, labels, img_name) in enumerate(tqdm(data)):
            inputs, labels = inputs.to(_device), labels.to(_device)
            output = network(inputs)
            tmp_outputs.append(output.cpu().numpy())
            tmp_labels.append(labels.cpu().numpy())
            tmp_img_names.append(img_name)

    outs = np.concatenate(tmp_outputs)
    labs = np.concatenate(tmp_labels).ravel()
    names = np.concatenate(tmp_img_names).ravel()

    for idx, name in enumerate(names):
        og_image_name = name.split("_")[0]
        if og_image_name in embeddings:
            embeddings[og_image_name].append(outs[idx])
        else:
            embeddings[og_image_name] = [outs[idx]]

    for key, val in tqdm(embeddings.items()):
        save_path = os.path.join(
                        base_path,
                        "embeddings-efficientnet-train",
                        f"{key}.pt"
                    )

        torch.save(
            torch.tensor(np.array(val)),
            save_path
        )

        # print(f'saved to: {save_path}')


def channel_copy(data: torch.tensor):
    return data.repeat(3, 1, 1)


if __name__ == '__main__':

    with wandb.init(project='DRAC2022_predictions') as run:
        artifact = run.use_artifact('markuskarner/DRAC2022/model_fanciful-sweep-5:v2', type='model')
        artifact_dir = artifact.download()


        TASK = 'c'  # for now only b and c work (classification)
        # "Classification B Quality" "Classification C Grading"
        TASK_DESC = "Classification C Grading"  # for logging only
        DATA_ROOT = "/system/user/publicdata/dracch/"  # "/Users/markus/Downloads/DRAC2022/"

        # Prepare paths
        if TASK == 'b':
            base_path = DATA_ROOT + "B. Image Quality Assessment"
            x_train_raw_path = base_path + "/patches_train/"
            y_train_raw_path = base_path + "/patches_train/annotation.csv"
        elif TASK == 'c':
            base_path = DATA_ROOT + "C. Diabetic Retinopathy Grading"
            x_train_raw_path = base_path + "/patches_train/"
            y_train_raw_path = base_path + "/patches_train/annotation.csv"
        else:
            raise Exception("Only Tasks b and c allowed!")

        dataloader_params = {
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 8,
            "pin_memory": True
        }

        mean = torch.load(base_path + "mean.pt")
        std = torch.load(base_path + "std.pt")

        patch_transforms = transforms.Compose([
            channel_copy,
            transforms.Resize((386, 386)),
            transforms.Normalize(
                mean=mean,
                std=std
            )])

        data_set = DRACPatchDataset(
            annotations_file=y_train_raw_path,
            img_dir=x_train_raw_path,
            transform=patch_transforms
        )

        data_loader = DataLoader(
            data_set,
            **dataloader_params
        )

        model = init_model(model="EfficientNet_B0", dropout=0)
        model_name = 'model_fanciful-sweep-5_30'

        model.load_state_dict(torch.load(artifact_dir + f"/{model_name}.pth"))
        model.eval()

        device = torch.device("cuda:0")
        model.classifier = Identity()
        model.to(device)

        calc_embeddings(
            network=model,
            data=data_loader
        )

