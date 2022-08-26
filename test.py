import pandas as pd
import numpy as np

DATA_ROOT = "/system/user/publicdata/dracch/"  # "/Users/markus/Downloads/DRAC2022/"

base_path = DATA_ROOT + "B. Image Quality Assessment/"
x_train_raw_path = base_path + "/1. Original Images/a. Training Set/"
y_train_raw_path = base_path + "/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv"

labels = pd.read_csv(y_train_raw_path)

#print(labels)

#print(labels.iloc[0, 1])


def to_binary(x):
    """
    Use to convert labels dataframe to binary
    :param x: original label
    :return: 3 values for 0vs1, 0vs2 and 1vs2. In the case of no comparison it returns None
    """

    if x == 0:
        return 0, 0, None
    elif x == 1:
        return 1, None, 0
    elif x == 2:
        return None, 1, 1
    else:
        raise Exception("Only three classes supported!")


labels_new = labels.copy()
labels_new['0vs1'], labels_new['0vs2'], labels_new['1vs2'] = zip(*labels_new["image quality level"].map(to_binary))

#print(labels_new)
#print(labels_new['0vs1'] * 1)


lol = labels_new.iloc[2:10, -3:]

test_array = np.random.random((5, 3))

print(test_array)
print(test_array[:, 0], test_array[:, 1], test_array[:, 2])
print("------------------")
class_0 = (1 - test_array[:, 0]) + (1 - test_array[:, 1])
class_1 = (test_array[:, 0]) + (1 - test_array[:, 2])
class_2 = test_array[:, 1] + test_array[:, 2]

scores = np.vstack((class_0, class_1, class_2)).T

print(scores)


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


print(softmax(scores))
print(np.sum(softmax(scores), axis=1))

exit()
