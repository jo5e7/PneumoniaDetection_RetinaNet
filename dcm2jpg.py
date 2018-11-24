import os
import pydicom

import numpy
from PIL.Image import fromarray

in_path = '/home/jdmaestre/PycharmProjects/Pneumonia_dataset/train'
out_path = '/home/jdmaestre/PycharmProjects/Pneumonia_dataset/jpg_train'

directory = os.fsencode(in_path)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".dcm"):
        #print(os.path.join(directory, file))
        image_in = in_path+'/'+filename
        print(image_in)
        ds = pydicom.read_file(image_in)
        im = fromarray(ds.pixel_array)



        filename = filename[:-4]
        image_out = out_path+'/'+filename
        im.save(image_out+'.jpg')
        continue
    else:
        continue