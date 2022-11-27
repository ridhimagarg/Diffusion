import pandas as pd
import shutil
import os

data = pd.read_csv("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert-v1.0-small/train.csv")

print(len(data))

print(data["Pleural Effusion"].count())

print(data["Pleural Effusion"].value_counts())

print(data["Pleural Effusion"].isna().sum())

filtered_data = data[(data["Pleural Effusion"] == 1) | (data["Pleural Effusion"] == 0)]
print(filtered_data)
print(len(filtered_data))

# os.mkdir("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert")
# os.mkdir(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert", "diseased"))
# os.mkdir("CheXpert/healthy")
# os.mkdir(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert", "healthy"))

# for row in filtered_data.iterrows():
#     # print(row)
#     if int(row[1]["Pleural Effusion"]) == 1:
#         name = row[1]["Path"].split("/")
#         shutil.copy(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset" ,row[1]["Path"]), os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert", "diseased_" + name[2]+ name[3]+ name[4]))
#     if int(row[1]["Pleural Effusion"]) == 0:
#         name = row[1]["Path"].split("/")
#         shutil.copy(os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset" ,row[1]["Path"]), os.path.join("/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpert", "healthy_" + name[2]+ name[3]+ name[4]))

