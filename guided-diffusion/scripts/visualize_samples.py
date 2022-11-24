import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

path = "../samples/openai-2022-09-17-11-08-57-608453"
array = np.load(os.path.join(path, "samples_100x256x256x3.npz"))

print(array.files)

samples = array["arr_0"]

print(samples)

print(samples[0].shape)

for img_idx in range(samples.shape[0]):

    filename = "sample_" + str(img_idx) + ".png"
    cv2.imshow("test", samples[img_idx])
    cv2.waitKey(0) 
  
    #closing all open windows 
    cv2.destroyAllWindows() 
    cv2.imwrite(os.path.join(path, filename), samples[img_idx])
