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

plt.savefig("traintime_modelscratch1.png")

plt.clf()

## Model-4 (/mount/arbeitsdaten/mudcat/Resources/Multimedia-Commons/dataset/CheXpertResults/classifiersample/openai-2022-12-17-22-12-27-510091_5000samples5attemptdiffusiononlyclassifierscale10timestemp250)
## FID (10 classifier scale): 105.00, sfid 90.81, xrV FID, 6.23

## Model-5(BIGGANMULT) - fid (96.24), sfid(105.4), xrv(4.83)

## Model-5 

## need to reeat for this config
## All on samples 5000, classifier scale= #Model1 -: 2classes_3attempt, for classifier - 2classes_3attempt
## Created the sampling for 10k, 20k,40k
time_steps = [10000, 20000, 40000, 85000]
fid_scores = [184.479, 144.39, 119.71, 65.95] ## all are nicely calculated expect for 10k
fid_scores_xrv = [10.75, 10.33, 4.94, 4.87] ## all are nicely calculated expect for 10k

plt.plot(time_steps, fid_scores, linestyle='--', marker='o', color='b', label="Inception FID(Model-1)")
plt.plot(time_steps, fid_scores_xrv, linestyle='--', marker='o', color='tab:pink', label="XRV FID(Model-1)")
plt.xlabel("Training diffusion time (iterations) fixing classifier steps 85000")
plt.ylabel("FID")
plt.legend()

# dataf = pd.DataFrame(np.c_[time_steps, fid_scores])
# print(dataf)
# ax = sns.lineplot(data=dataf, marker= 'h', markersize=10)
plt.savefig("traintime_fid1.png")

plt.clf()

# sampling_steps 
## For model-1 itself
sampling_steps = [250, 1000]
# classifier_scale = [1, 10]
fid_scores_scale1 = [65.95, 50.05]
fid_scores_scale5 = [64.52, 50.03] ## for 1000 timesteps (not calculated very well still running the generating samples script)
fid_scores_scale10 = [52.25 , 45.25] 
fid_scores_scale12 = [65.11, 49.08]    ## for 1000 timesteps (not calculated very well still running the generating samples script)
fid_scores_scale15 = [66.02, 50.06] ## for 1000 timesteps (not calculated at all)
fid_scores_xrv_scale1 = [4.87, 4.32]
fid_scores_xrv_scale5 = [3.43, 3.67] ## for 1000 timesteps (not calculated very well still running the generating samples script)
fid_scores_xrv_scale10 = [4.50, 3.09]
fid_scores_xrv_scale12 = [3.38, 3.78] ## for 1000 timesteps (not calculated very well still running the generating samples script)
fid_scores_xrv_scale15 = [3.42, 3.78] ## for 1000 timesteps (not calculated at all)

## Running for classifier with timestemo 250 scale 1,5(784139),10(565034),12(238406),15()
# Running for classifier with timestemp 1000 scale 1,5(),10(295650),12(),15()

# legendFig = plt.figure("Legend plot")

plt.plot(sampling_steps, fid_scores_scale1, linestyle='--', marker='o', color='b', label="Inception FID classifier scale=1")
plt.plot(sampling_steps, fid_scores_scale5, linestyle='--', marker='o', color='tab:brown', label="Inception FID classifier scale=5")
plt.plot(sampling_steps, fid_scores_scale10, linestyle='--', marker='o', color='tab:orange', label="Inception FID classifier scale=10")
plt.plot(sampling_steps, fid_scores_scale12, linestyle='--', marker='o', color='darkblue', label="Inception FID classifier scale=12")
plt.plot(sampling_steps, fid_scores_scale15, linestyle='--', marker='o', color='slateblue', label="Inception FID classifier scale=15")
plt.plot(sampling_steps, fid_scores_xrv_scale1, linestyle='--', marker='o', color='tab:pink', label="XRV FID classifier scale=1")
plt.plot(sampling_steps, fid_scores_xrv_scale5, linestyle='--', marker='o', color='salmon', label="XRV FID classifier scale=5")
plt.plot(sampling_steps, fid_scores_xrv_scale10, linestyle='--', marker='o', color='tab:purple', label="XRV FID classifier scale=10")
plt.plot(sampling_steps, fid_scores_xrv_scale12, linestyle='--', marker='o', color='goldenrod', label="XRV FID classifier scale=12")
plt.plot(sampling_steps, fid_scores_xrv_scale15, linestyle='--', marker='o', color='cyan', label="XRV FID classifier scale=15")
plt.xlabel("sampling steps")
plt.ylabel("FID")
plt.legend()
# plt.legend(loc='upper right')

plt.savefig("samplingtime_diffclassifierscales_fid1.png")
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
# fid_sfid_xrv_classifier_scale_5 = 29.13, 76.32, 1.46
# fid_sfid_xrv_classifier_scale_12 = 30.24, 77.61, 1.63
## Running for classifier scale 5, 12 for 18k diffusion.

## classwise for each timesteps
finetuning_timesteps =[18000, 28000, 36000]
fid_class_0 = [26.37,40.78, 32.15]
sfid_class_0 = [109.10,109.06, 103.84]
fid_class_1 = [52.92,43.99, 30.59]
sfid_class_1 = [126.09,106.08, 105.46]
 
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

plt.savefig("samplingtime_fid_finetunemodel1.png")
plt.clf()


## Finetuned model only for 250 timesteps.
## Classwise FID for all 3 timesteps for the finetuned model. - done created an array for it.
## For finetuned model, lets check for 18k training time model, with different classifier scales. - running.