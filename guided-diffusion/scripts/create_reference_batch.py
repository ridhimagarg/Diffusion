import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from PIL import Image

# path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-11-25-22-09-07-713511-1000classes_script"
path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert"
# array = np.load(os.path.join(path, "VIRTUAL_imagenet128_labeled.npz"))
# foldername = "VIRTUAL_imagenet128_labeled"

files = os.listdir(path)

files_selected = random.choices(files, k=500)
# print(files_selected)

images = []

for img_file in files_selected:
    
    np_array = cv2.imread(os.path.join(path, img_file))
    # np_array = Image.open(os.path.join(path, img_file))
    # print(np_array.resize(256,256).shape)
    np_array = cv2.resize(np_array, (256,256), interpolation = cv2.INTER_AREA)
    images.append(np_array)

# print(images)
print(np.array(images).shape)

np.savez("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/evaluations_base/500_256_256_3_reference_batch.npz" ,np.array(images))


