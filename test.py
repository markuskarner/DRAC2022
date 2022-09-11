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
