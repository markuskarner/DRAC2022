import pandas as pd
import numpy as np
from helpers import prepare_classification_dataset
import glob
import os
import pandas as pd

DATA_ROOT = "/system/user/publicdata/dracch/"  # "/Users/markus/Downloads/DRAC2022/"

base_path = DATA_ROOT + "C. Diabetic Retinopathy Grading"
x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
y_train_raw_path = base_path + "/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv"
x_train_patches_path = base_path + "/patches_train/"
x_train_embeddings_path = base_path + "/embeddings-efficientnet-train/"

def create_annotation_file_for_patches(patches_path, labels_path):
    def image_name_from_patch(patch_id: str):
        return patch_id.split('_')[0] + '.png'

    labels = pd.read_csv(labels_path)

    files = [os.path.basename(x) for x in glob.glob(patches_path + "*.png",)]
    df = pd.DataFrame(files)

    df['image name'] = df[0].map(image_name_from_patch)
    annotation_df = pd.merge(df, labels, on='image name', how='inner')

    annotation_df.drop(columns='image name', inplace=True)

    annotation_df.to_csv(patches_path + 'annotation.csv', index=False)


# create_annotation_file_for_patches(x_train_patches_path, y_train_raw_path)

dataloader_train, dataloader_valid, train_target = prepare_classification_dataset(base_path,
                                                                                  x_train_embeddings_path,
                                                                                  y_train_raw_path,
                                                                                  8,
                                                                                  model='EfficientNet_B0',
                                                                                  task='c',
                                                                                  use_for_attention=True)

ys = []
for batch_idx, (x, y) in enumerate(dataloader_train):
    print(x.shape, y)

exit()
