import torchxrayvision as xrv
import torch
import numpy as np

model = xrv.models.DenseNet(weights="densenet121-res224-all")
# arr = np.load("../../Results/openai-2022-12-20-15-56-38-239194/samples_100x256x256x3.npz")["arr_0"]
# print(arr.shape)
# arr1 = arr.reshape((arr.shape[0], arr.shape[3], arr.shape[1], arr.shape[2]))
# # arr = arr.mean(1)[None, ...]
# arr = np.mean(arr1, axis=1).reshape((arr.shape[0], 1, arr.shape[1], arr.shape[2]))
# # print(model.features(torch.from_numpy(arr).float()).shape)

# print(np.mean(model.features2(torch.from_numpy(arr).float()).detach().numpy(), axis=0).shape)

# print(np.cov(model.features2(torch.from_numpy(arr).float()).detach().numpy(), rowvar=False).shape)

def densenet_features(input):

    # print("input", input.shape)
    # input1 = input.reshape((input.shape[0], input.shape[3], input.shape[1], input.shape[2]))
    # input = xrv.datasets.normalize(input, 255)
    input1 = torch.mean(input, axis=1, keepdim=True)
    # print("input1", input1.shape)

    return model.features2(input1.float()).detach().numpy()


