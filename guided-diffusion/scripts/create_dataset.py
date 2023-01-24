import pandas as pd
import shutil
import os
import numpy as np

data = pd.read_csv("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert-v1.0-small/train.csv")

print(len(data))

## ------------------------- Only for Pleural Effusion ---------------------- ##

# print(data["Pleural Effusion"].count())

# print(data["Pleural Effusion"].value_counts())

# print(data["Pleural Effusion"].isna().sum())

# filtered_data = data[(data["Pleural Effusion"] == 1) | (data["Pleural Effusion"] == 0)]
# print(filtered_data)
# print(len(filtered_data))

# os.mkdir("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertValidation")
# # os.mkdir(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert", "diseased"))
# # os.mkdir("CheXpert/healthy")
# # os.mkdir(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert", "healthy"))

# for row in filtered_data.iterrows():
#     # print(row)
#     if int(row[1]["Pleural Effusion"]) == 1:
#         name = row[1]["Path"].split("/")
#         shutil.copy(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset" ,row[1]["Path"]), os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertValidation", "diseased_" + name[2]+ name[3]+ name[4]))
#     if int(row[1]["Pleural Effusion"]) == 0:
#         name = row[1]["Path"].split("/")
#         shutil.copy(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset" ,row[1]["Path"]), os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertValidation", "healthy_" + name[2]+ name[3]+ name[4]))


##------------------------------- All Pathalogy data -------------------------- ##

pathologies = data.columns[6:]

print("Pathologies", pathologies)

for row in data.iterrows():
    for patho in pathologies:
        if not np.isnan(row[1][patho]):   
            if int(row[1][patho]) == 1:
                name = row[1]["Path"].split("/")
                print(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertAllPathology/training1", str(patho).replace(" ", "") + "diseased_" + name[2]+ name[3]+ name[4]))
                shutil.copy(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset" ,row[1]["Path"]), os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertAllPathology/training1", str(patho).replace(" ","") + "diseased_" + name[2]+ name[3]+ name[4]))
            if int(row[1][patho]) == 0:
                name = row[1]["Path"].split("/")
                print(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertAllPathology/training1", str(patho).replace(" ", "") + "healthy_" + name[2]+ name[3]+ name[4]))
                shutil.copy(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset" ,row[1]["Path"]), os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertAllPathology/training1", str(patho).replace(" ", "") + "healthy_" + name[2]+ name[3]+ name[4]))
        

