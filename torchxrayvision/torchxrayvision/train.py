from datasets import CheX_Dataset
from torch.utils.data import DataLoader
from models import DenseNet
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4

train_ds = CheX_Dataset()

train_loader = DataLoader(train_ds,
                            batch_size=8)


model = DenseNet().to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss() ## cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

mini_batch_losses = []

for X in train_loader:

    # print(X["lab"])
    print(X["img"].shape)
    X["img"] = X["img"].to(DEVICE)
    X["lab"] = X["lab"].to(DEVICE)

    model.train()
    predictions = model(X["img"])
    
    loss = loss_fn(predictions, X["lab"])

    mini_batch_losses.append(loss)

    loss.backward()

    optimizer.zero_grad()
    optimizer.step()
    
print("Training loss:", np.mean(mini_batch_losses))



