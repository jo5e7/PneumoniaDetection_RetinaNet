from shutil import copy
import pandas as pd
import os
import pydicom

import numpy
from PIL.Image import fromarray


# Change
import matplotlib.pyplot as plt
plt.plot([1,2])
plt.plot([2,3])
plt.plot([3,4])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Epoch losses")
plt.show()
#Change

original_df = pd.read_csv("/home/jdmaestre/PycharmProjects/Pneumonia_dataset/stage_2_train_labels.csv")
out_test = []
out_train = []


out_path_test = '/home/jdmaestre/PycharmProjects/Pneumonia_dataset_synthetic/synthetic_test/'
out_path_train = '/home/jdmaestre/PycharmProjects/Pneumonia_dataset_synthetic/synthetic_train/'
out_path = ''

for index, row in original_df.iterrows():
    file = row[0]
    print(index, file)
    obj = []


    for n in range(0,6):
        obj.append(row[n])
        pass


    if (index < 3000):
        out_path = out_path_test
        out_test.append(obj)
        pass
    else:
        out_path = out_path_train
        out_train.append(obj)
        pass

    copy(file, out_path)
    pass

out_test = pd.DataFrame(out_test)
out_test.to_csv("/home/jdmaestre/PycharmProjects/Pneumonia_dataset_synthetic/synthetic_test_set.csv")

out_train = pd.DataFrame(out_train)
out_train.to_csv("/home/jdmaestre/PycharmProjects/Pneumonia_dataset_synthetic/synthetic_train_set.csv")