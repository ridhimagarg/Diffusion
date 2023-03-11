import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from PIL import Image

# path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-11-25-22-09-07-713511-1000classes_script"
path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert"
path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertAllPathology/training1"
# array = np.load(os.path.join(path, "VIRTUAL_imagenet128_labeled.npz"))
# foldername = "VIRTUAL_imagenet128_labeled"
# path

classes = {'Atelectasisdiseased': 0, 'Atelectasishealthy': 1, 'Cardiomegalydiseased': 2, 'Cardiomegalyhealthy': 3, 'Consolidationdiseased': 4, 'Consolidationhealthy': 5, 'Edemadiseased': 6, 'Edemahealthy': 7, 'EnlargedCardiomediastinumdiseased': 8, 'EnlargedCardiomediastinumhealthy': 9, 'Fracturediseased': 10, 'Fracturehealthy': 11, 'LungLesiondiseased': 
12, 'LungLesionhealthy': 13, 'LungOpacitydiseased': 14, 'LungOpacityhealthy': 15, 'PleuralEffusiondiseased': 16, 'PleuralEffusionhealthy': 17, 'PleuralOtherdiseased': 18, 'PleuralOtherhealthy': 19, 'Pneumoniadiseased': 20, 'Pneumoniahealthy': 21, 'Pneumothoraxdiseased': 22, 'Pneumothoraxhealthy': 23, 'SupportDevicesdiseased': 24, 'SupportDeviceshealthy': 25}


images = []
images_labels = []
for class_, class_index in classes.items():

    files = os.listdir(path)

    files_selected = random.choices([f for f in files if f.startswith(class_)], k=192)
    # print(files_selected)

    for img_file in files_selected:
        
        np_array = cv2.imread(os.path.join(path, img_file))
        # np_array = Image.open(os.path.join(path, img_file))
        # print(np_array.resize(256,256).shape)
        np_array = cv2.resize(np_array, (256,256), interpolation = cv2.INTER_AREA)
        images.append(np_array)
    images_labels.extend([class_index]* len(files_selected))
    # print(images_labels)

    # print(images)


files_selected = random.choices([f for f in files if f.startswith("Atelectasisdiseased")], k=8)
    # print(files_selected)

for img_file in files_selected:
    
    np_array = cv2.imread(os.path.join(path, img_file))
    # np_array = Image.open(os.path.join(path, img_file))
    # print(np_array.resize(256,256).shape)
    np_array = cv2.resize(np_array, (256,256), interpolation = cv2.INTER_AREA)
    images.append(np_array)

images_labels.extend([0]* len(files_selected))
# print(images_labels)

print(np.array(images).shape)
    
np.savez("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/evaluations_base/5000_multilabel_256_256_3_reference_batch.npz" ,np.array(images), np.array(images_labels))