import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

orig_csv_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert-v1.0-small/train.csv"
# array_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-14-23-12-24-295650_5000samples3attemptclassifierscale10timespacedim1000/samples_5000x256x256x3.npz"
# array_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-05-17-46-00-565034_5000samples3attempt10classifierscalelatermodels/samples_5000x256x256x3.npz"
# array_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-17-22-12-27-510091_5000samples5attemptdiffusiononlyclassifierscale10timestemp250/samples_5000x256x256x3.npz"
# save_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert-fake/attempt3"


# array_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2023-02-03-22-25-29-146934_diffusion_finetune_36k_timestep250/samples_5000x256x256x3.npz"
# save_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert-fake/attempt4"  

array_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-05-17-46-00-565034_5000samples3attempt10classifierscalelatermodels/samples_5000x256x256x3.npz"
save_path = "/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert-fake/attempt5"


array = np.load(array_path)

# print(save_path.rsplit("/", 2)[0])

print(array.files)

# print(array["arr_1"])

df = pd.read_csv(orig_csv_path)

print(df.columns)


new_df = pd.DataFrame(columns=df.columns.to_list())

for index, arr in enumerate(array["arr_0"]):

    if not os.path.exists(os.path.join(save_path, "patient_" + str(index))):
        os.mkdir(os.path.join(save_path, "patient_" + str(index)))
        cv2.imwrite(os.path.join(save_path, "patient_" + str(index), "image.jpg"), arr)
    print("hello..")

    # print(os.path.join(save_path, "patient_" + str(index), "image.jpg"))

    new_row = pd.DataFrame([{"Path": os.path.join(save_path.rsplit("/", 2)[1],  save_path.rsplit("/", 1)[1], "patient_"+ str(index) , "image.jpg"),
               "Sex" : np.nan , "Age": np.nan, "Frontal/Lateral" : np.nan, "AP/PA": np.nan, "No Finding": np.nan,
       "Enlarged Cardiomediastinum": np.nan, "Cardiomegaly": np.nan, "Lung Opacity": np.nan,
       "Lung Lesion": np.nan, "Edema": np.nan, "Consolidation": np.nan, "Pneumonia": np.nan, "Atelectasis": np.nan,
       "Pneumothorax" : np.nan, "Pleural Effusion" : array["arr_1"][index], 'Pleural Other': np.nan, 'Fracture': np.nan,'Support Devices': np.nan}])
    new_df = pd.concat([new_df, new_row])

print(new_df.loc[0,"Path"]) 

print(new_df)

new_df.to_csv(os.path.join(save_path, "test.csv"), index=False)

final_df = pd.concat([df, new_df])
print(len(final_df))

final_df.to_csv(os.path.join(save_path, "train.csv"), index=False)


    

