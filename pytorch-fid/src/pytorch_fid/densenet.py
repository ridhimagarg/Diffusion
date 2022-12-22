import torchxrayvision as xrv

model = xrv.models.DenseNet(weights="densenet121-res224-all")
print(model.we)
# outputs = model(img[None,...])