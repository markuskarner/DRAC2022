import wandb
import torch
import pandas as pd

from torch import nn

from helpers import prepare_transform, DracClassificationDatasetTest, DataLoader, init_model

if __name__ == "__main__":

    TASK = 'b'  # for now only b and c work (classification)
    TASK_DESC = "Classification B Quality"  # for logging only
    DATA_ROOT = "/system/user/publicdata/dracch/"  # "/Users/markus/Downloads/DRAC2022/"

    # Prepare base paths
    if TASK == 'b':
        base_path = DATA_ROOT + "B. Image Quality Assessment/"
    elif TASK == 'c':
        base_path = DATA_ROOT + "C. Diabetic Retinopathy Grading/"
    else:
        raise Exception("Only Tasks b and c allowed!")

    x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
    x_test_raw_path = base_path + "/1. Original Images/b. Testing Set/"

    transform = prepare_transform(base_path, x_train_raw_path, False)
    data_test = DracClassificationDatasetTest(x_test_raw_path, transform["test"])
    dataloader_test = DataLoader(data_test, batch_size=16, num_workers=8)

    with wandb.init() as run:
        artifact = run.use_artifact('markuskarner/DRAC2022/model_comic-sweep-1:v2', type='model')
        artifact_dir = artifact.download()

        model_name = "/model_comic-sweep-1_30.pth"

        model = init_model("ConvNeXt_tiny", 0.)
        model.load_state_dict(torch.load(artifact_dir + model_name))
        model.eval()

        device = torch.device("cuda:0")
        model.to(device)

        output_list = []

        for batch_idx, (x, img_id) in enumerate(dataloader_test):
            device = next(model.parameters()).device
            x = x.float().to(device)

            output = model(x)

            for o, i in zip(output, img_id):
                probs = nn.functional.softmax(o, dim=0)
                output_argmax = torch.argmax(o).item()
                output_list.append((i, output_argmax, probs[0].item(), probs[1].item(), probs[2].item()))

        df = pd.DataFrame(output_list, columns=['case', 'class', 'P0', 'P1', 'P2'])
        df.to_csv('/system/user/publicwork/student/karner/model_comic-sweep-1_30.csv', index=False)

        print(df.groupby(['class']).size())
