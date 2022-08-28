import pandas as pd
import numpy as np
from helpers import prepare_classification_dataset

DATA_ROOT = "/system/user/publicdata/dracch/"  # "/Users/markus/Downloads/DRAC2022/"

base_path = DATA_ROOT + "B. Image Quality Assessment/"
x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
y_train_raw_path = base_path + "/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv"

labels = pd.read_csv(y_train_raw_path)


dataloader_train, dataloader_valid, train_target = prepare_classification_dataset(base_path,
                                                                                  x_train_raw_path,
                                                                                  y_train_raw_path,
                                                                                  8,
                                                                                  model='ResNet50')

ys = []
for batch_idx, (x, y, y_bce, mask) in enumerate(dataloader_valid):
    ys.append(y.cpu().detach().numpy())

print(np.hstack(ys))

unique, counts = np.unique(np.hstack(ys), return_counts=True)
print(dict(zip(unique, counts)))

exit()
