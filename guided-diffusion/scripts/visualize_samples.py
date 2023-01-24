import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-20-15-56-38-239194"
# path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2023-01-21-12-38-04-126649"
path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-05-17-46-00-565034_5000samples3attempt10classifierscalelatermodels"
# path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/evaluations_base"
array = np.load(os.path.join(path, "samples_5000x256x256x3.npz"))
foldername = "samples_12x256x256x3"

print(array.files)

# print(array['arr_0'].shape)
print(array['arr_0'][10:20])
print(array["arr_1"][10:20])

# samples = array["arr_0"]

# print(samples.shape[0])

# print(samples[0].shape)

# os.mkdir(os.path.join(path, foldername))

# for img_idx in range(samples.shape[0]):

#     print("here")

#     filename = "sample_" + str(img_idx) + ".png"
#     # cv2.imshow("test", samples[img_idx])
#     # cv2.waitKey(0) 
  
#     # #closing all open windows 
#     # cv2.destroyAllWindows() 
#     print(os.path.join(path, foldername, filename))
    
#     cv2.imwrite(os.path.join(path, foldername, filename), samples[img_idx])
