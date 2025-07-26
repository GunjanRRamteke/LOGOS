import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


class MLP_model(nn.Module):

    def __init__(sf, in_channel, out_channel):
        super().__init__()

        sf.layers = nn.Sequential(
        nn.Linear(in_channel, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 200),
        nn.ReLU(),
        nn.Linear(200, 160),
        nn.ReLU(),
        nn.Linear(160, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, out_channel))

    def forward(sf,x):
        return(sf.layers(x))


def Predict_Pretrained(X_dat, O_len, filename):

    scaler = StandardScaler()
    scaler.fit(X_dat)
    X_dat = scaler.transform(X_dat)
    X_dat = torch.tensor(X_dat, dtype=torch.float32)

    model = MLP_model(len(X_dat[0]), O_len)
    saved_M = filename
    model.load_state_dict(torch.load(saved_M))
    model.eval()

    y_pred = model(X_dat)
    y_pred = y_pred.detach().numpy()

    return(y_pred)

