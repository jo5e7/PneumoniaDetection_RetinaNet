import os
import pydicom
import pandas as pd
import numpy
from PIL.Image import fromarray

df = pd.read_csv('/home/jdmaestre/PycharmProjects/stage_2_train_labels.csv')
df_compare = pd.read_csv('/home/jdmaestre/PycharmProjects/stage_2_detailed_class_info.csv')

normal_path = '/home/jdmaestre/PycharmProjects/output/Normal/'
lung_opacity_path = '/home/jdmaestre/PycharmProjects/output/Lung Opacity/'
lo_nn_path = '/home/jdmaestre/PycharmProjects/output/No Lung Opacity - Not Normal/'

for index, row in df.iterrows():
    file = ''
    filename = row['patientId']
    output_class = ''

    for index_c, row_c in df_compare.iterrows():
        if filename == row_c['patientId']:
            output_class = row_c['class']
            file = row_c['location']
            break
            pass
        pass

    if file.endswith(".dcm"):
        # print(os.path.join(directory, file))
        print(file)
        ds = pydicom.read_file(file)
        im = fromarray(ds.pixel_array)

        filename = filename + ".dcm"

        if (output_class == 'Normal'):
            image_out = normal_path + filename
            im.save(image_out + '.jpg')
            pass

        if (output_class == 'No Lung Opacity / Not Normal'):
            image_out = lo_nn_path + filename
            im.save(image_out + '.jpg')
            pass

        if (output_class == 'Lung Opacity'):
            image_out = lung_opacity_path + filename
            im.save(image_out + '.jpg')
            pass
        continue
    else:
        continue
    pass