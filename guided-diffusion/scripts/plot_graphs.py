import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

iterations = [50000, 60000, 80000, 100000]
time_taken_model1 = [10.04, 12.20, 16.16, 22]
iterations1 = [40000, 50000, 60000]
time_taken_model5 = [10.30, 12.12, 14]

plt.plot(iterations, time_taken_model1, linestyle='--', marker='o', color='b', label="Time(Model-1)")
plt.plot(iterations1, time_taken_model5, linestyle='--', marker='o', color='tab:pink', label="Time(Model-5)")
plt.xlabel("Iterations")
plt.ylabel("Time taken by model")
plt.legend()

plt.savefig("traintime_modelscratch.png")

plt.clf()



## need to reeat for this config
## All on samples 5000, classifier scale= #Model1 -: 2classes_3attempt, for classifier - 2classes_3attempt
## Created the sampling for 10k, 20k,40k
time_steps = [10000, 20000, 40000, 85000]
fid_scores = [184.479, 160.95, 110, 65.95]
fid_scores_xrv = [10.75, 8.25, 6, 4.87]

plt.plot(time_steps, fid_scores, linestyle='--', marker='o', color='b', label="Inception FID(Model-1)")
plt.plot(time_steps, fid_scores_xrv, linestyle='--', marker='o', color='tab:pink', label="XRV FID(Model-1)")
plt.xlabel("Training diffusion time (iterations) fixing classifier steps 85000")
plt.ylabel("FID")
plt.legend()

# dataf = pd.DataFrame(np.c_[time_steps, fid_scores])
# print(dataf)
# ax = sns.lineplot(data=dataf, marker= 'h', markersize=10)
plt.savefig("traintime_fid.png")

plt.clf()

# sampling_steps 
## For model-1 itself
sampling_steps = [250, 1000]
# classifier_scale = [1, 10]
fid_scores_scale1 = [65.95, 50.05]
fid_scores_scale10 = [52.25 , 45.25]
fid_scores_scale12 = [65.11]    
fid_scores_xrv_scale1 = [4.87, 4.32]
fid_scores_xrv_scale10 = [4.50, 3.09]

## Running for classifier with timestemo 250 scale 1,5(),10(565034),12(238406),15()
# Running for classifier with timestemp 1000 scale 1,5(),10(295650),12(),15()

# legendFig = plt.figure("Legend plot")

plt.plot(sampling_steps, fid_scores_scale1, linestyle='--', marker='o', color='b', label="Inception FID classifier scale=1")
plt.plot(sampling_steps, fid_scores_scale10, linestyle='--', marker='o', color='tab:orange', label="Inception FID classifier scale=10")
plt.plot(sampling_steps, fid_scores_xrv_scale1, linestyle='--', marker='o', color='tab:pink', label="XRV FID classifier scale=1")
plt.plot(sampling_steps, fid_scores_xrv_scale10, linestyle='--', marker='o', color='tab:purple', label="XRV FID classifier scale=10")
plt.xlabel("sampling steps")
plt.ylabel("FID")
plt.legend()
# plt.legend(loc='upper right')

plt.savefig("samplingtime_fid.png")
plt.clf()

# legendFig.legend([line1, line2], ["y=log(x)", "y=sin(x)"], loc='center')


# fig, ax = plt.subplots(figsize=(4,3))

# fig = plt.figure("Line plot")
# legendFig = plt.figure("Legend plot")

# ax = fig.add_subplot(111)

# # Plotting with label
# line_1 = ax.plot(sampling_steps, fid_scores_scale1, linestyle='--', marker='o', color='b', label="Inception FID classifier scale=1")
# line_2 = ax.plot(sampling_steps, fid_scores_scale10, linestyle='--', marker='o', color='tab:orange', label="Inception FID classifier scale=10")
# # line_3 = ax.plot(sampling_steps, fid_scores_xrv_scale1, linestyle='--', marker='o', color='tab:pink', label="XRV FID classifier scale=1")
# # line_4 = ax.plot(sampling_steps, fid_scores_xrv_scale10, linestyle='--', marker='o', color='tab:purple', label="XRV FID classifier scale=10")


# # Lines to show on legend and their labels
# # lines = [line_1, line_2, line_3, line_4]
# # labels = [i.get_label() for i in lines]
# # print(labels)
# legendFig.legend([line_1, line_2], ["y=log(x)", "y=sin(x)"], loc='center')
# legendFig.savefig('legend.png')


#================================

# fid_sfid_xrv_classifier_scale_10 = 30, 77.72, ## can check for report if decreasing the classifier scale improves the accuracy.
## Running for classifier scale 5, 12 for 18k diffusion.

## classwise for each timesteps
finetuning_timesteps =[18000, 28000, 36000]
fid_class_0 = [26.37,40.78]
sfid_class_0 = [109.10,109.06]
fid_class_1 = [52.92,43.99]
sfid_class_1 = [126.09,106.08]
 
fig = plt.figure()      
# ax = fig.add_subplot(111)
finetuning_timesteps =[18000, 28000, 36000]
fid = [30.13, 29.78, 20.41]
sfid = [75.94, 69.08, 66.18]
fid_xrv = [1.3508,1.354,0.7101]

## Running for classifier with timestemo 250 all with classifier scale 1.0 trainingsteps 18k(471232, 229058), 28k(026543), 36k(301174, 886043) - All calulated nicely.

plt.plot(finetuning_timesteps, fid, linestyle='--', marker='o', color='b', label="Inception FID")
plt.plot(finetuning_timesteps, sfid, linestyle='--', marker='o', color='tab:orange', label="Inception sFID")
plt.plot(finetuning_timesteps, fid_xrv, linestyle='--', marker='o', color='tab:pink', label="XRV FID")
# plt.xticks(range(len(finetuning_timesteps)) , finetuning_timesteps)

# for i, v in enumerate(fid):
#     ax.text(i, v+25, "%d" %v, ha="center")

for i in range(len(finetuning_timesteps)):
    plt.annotate(str(fid[i]), xy=(finetuning_timesteps[i], fid[i]))
    plt.annotate(str(sfid[i]), xy=(finetuning_timesteps[i], sfid[i]))
    plt.annotate(str(fid_xrv[i]), xy=(finetuning_timesteps[i], fid_xrv[i]))
# plt.ylim(-10, 595)    
plt.xlabel("train iterations")
plt.ylabel("FID")
plt.legend()

plt.savefig("samplingtime_fid_finetunemodel.png")
plt.clf()


## Finetuned model only for 250 timesteps.
## Classwise FID for all 3 timesteps for the finetuned model. - done created an array for it.
## For finetuned model, lets check for 18k training time model, with different classifier scales. - running.