import wandb
import torch
import pandas as pd
from tqdm import tqdm

from torch import nn

from helpers import prepare_transform, DracClassificationDatasetTest, DataLoader, init_model

if __name__ == "__main__":

    TASK = 'c'  # for now only b and c work (classification)
    # "Classification B Quality" "Classification C Grading"
    TASK_DESC = "Classification C Grading"  # for logging only
    DATA_ROOT = "/system/user/publicdata/dracch/"  # "/Users/markus/Downloads/DRAC2022/"
    MODEL = "EfficientNet_B0"

    # Prepare base paths
    if TASK == 'b':
        base_path = DATA_ROOT + "B. Image Quality Assessment/"
        id_to_label = {0: 'Poor quality level', 1: 'Good quality level', 2: 'Excellent quality level'}
    elif TASK == 'c':
        base_path = DATA_ROOT + "C. Diabetic Retinopathy Grading/"
        id_to_label = {0: 'Normal', 1: 'NPDR', 2: 'PDR'}
    else:
        raise Exception("Only Tasks b and c allowed!")

    x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
    x_test_raw_path = base_path + "/1. Original Images/b. Testing Set/"

    transform = prepare_transform(base_path, x_train_raw_path, model=MODEL)
    data_test = DracClassificationDatasetTest(x_test_raw_path, transform["test"])
    dataloader_test = DataLoader(data_test, batch_size=8, num_workers=8)

    with wandb.init(project='DRAC2022_predictions') as run:
        artifact = run.use_artifact('markuskarner/DRAC2022/model_sandy-sweep-31:v0', type='model')
        artifact_dir = artifact.download()

        # create an artifact for all raw test images
        raw_data_at = wandb.Artifact('raw_data_1000', type="raw_data")

        # create a table with columns we want to track/compare
        predictions_table = wandb.Table(columns=['case', 'image','class', 'P0', 'P1', 'P2'])

        model_name = 'model_sandy-sweep-31_10'

        model = init_model(MODEL, 0.)
        model.load_state_dict(torch.load(artifact_dir + f"/{model_name}.pth"))
        model.eval()

        device = torch.device("cuda:0")
        model.to(device)

        output_list = []

        for batch_idx, (x, img_id) in enumerate(dataloader_test):
            device = next(model.parameters()).device
            x = x.float().to(device)

            output = model(x)

            for o, i, _x in zip(output, img_id, x):

                y_sigmoid = torch.sigmoid(o)

                class_0 = (1 - y_sigmoid[0]) + (1 - y_sigmoid[1])
                class_1 = (y_sigmoid[0]) + (1 - y_sigmoid[2])
                class_2 = y_sigmoid[1] + y_sigmoid[2]

                y_hat_before_softmax = torch.hstack((class_0, class_1, class_2)).T

                probs = nn.functional.softmax(y_hat_before_softmax, 0)

                output_argmax = torch.argmax(y_hat_before_softmax).item()
                output_list.append((i, output_argmax, probs[0].item(), probs[1].item(), probs[2].item()))

                # wandb.log({"prediction": [wandb.Image(_x, caption=output_argmax)]})

        df = pd.DataFrame(output_list, columns=['case', 'class', 'P0', 'P1', 'P2'])
        df.to_csv(f'/system/user/publicwork/student/karner/AILS_Students@JKU_{model_name}.csv', index=False)

        for i in tqdm(range(len(df))):
            image_name = df.loc[i]['case']
            label = df.loc[i]['class']
            P0 = df.loc[i]['P0']
            P1 = df.loc[i]['P1']
            P2 = df.loc[i]['P2']
            full_path = x_test_raw_path + image_name

            # Append each example as a new row in Table.
            predictions_table.add_data(image_name, wandb.Image(full_path), id_to_label[label], P0, P1, P2)

        raw_data_at.add(predictions_table,'predictions')
        run.log_artifact(raw_data_at)

        predictions_csv = wandb.Artifact(f'csv_for_upload', type='csv')
        predictions_csv.add_file(f'/system/user/publicwork/student/karner/AILS_Students@JKU_{model_name}.csv')
        wandb.log_artifact(predictions_csv)

        print(df.groupby(['class']).size())
