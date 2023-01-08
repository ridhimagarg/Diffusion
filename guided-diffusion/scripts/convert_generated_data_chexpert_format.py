import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

orig_csv_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert-v1.0-small/train.csv"
path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-20-15-56-38-239194"
# path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/evaluations_base"
array = np.load(os.path.join(path, "samples_100x256x256x3.npz"))
foldername = "samples_100x256x256x3"

print(array.files)

print(array["arr_1"])

data = pd.read_csv(orig_csv_path)

print(data.columns)

