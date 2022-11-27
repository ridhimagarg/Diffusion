import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-11-25-22-09-07-713511-1000classes_script"
path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/evaluations_base"
array = np.load(os.path.join(path, "VIRTUAL_imagenet128_labeled.npz"))

print(array.files)

# print(array['arr_0'].shape)
print(array['arr_0'][0])

# samples = array["arr_0"]

# print(samples)

# print(samples[0].shape)

# for img_idx in range(samples.shape[0]):

#     filename = "sample_" + str(img_idx) + ".png"
#     cv2.imshow("test", samples[img_idx])
#     cv2.waitKey(0) 
  
#     #closing all open windows 
#     cv2.destroyAllWindows() 
#     cv2.imwrite(os.path.join(path, filename), samples[img_idx])
