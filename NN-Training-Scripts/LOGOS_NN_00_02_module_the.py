import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import copy
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
        nn.Dropout(0.2),
        nn.Linear(200, 160),
        nn.ReLU(),
        nn.Dropout(0.2),
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

class NeuralNetwork_MAIN():

    def __init__(sf):
        pass

    def NeuralNetwork(sf, I_grp, prop, filename):

        X_train, X_test, y_train, y_test = train_test_split(I_grp, prop, train_size=0.7, shuffle=True)

        sf.scaler = StandardScaler()
        sf.scaler.fit(I_grp)
        X_train = sf.scaler.transform(X_train)
        X_test  = sf.scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        learning_rate = 0.001
        n_epochs = 150
        batch_size = 50

        sf.model = MLP_model(len(X_train[0]), len(y_train[0]))
        optimizer = optim.Adam(sf.model.parameters(), learning_rate)
        loss_fn = nn.MSELoss()

        batch_start = torch.arange(0, len(X_train), batch_size)
        best_mse = np.inf
        best_weights = None
        history = []

        '''   Training Loop  '''

        for epoch in range(n_epochs):
            sf.model.train()
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            batch_start = torch.arange(0, len(X_train_shuffled), batch_size)

            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                 bar.set_description(f"Epoch {epoch}")
                 for start in bar:
                     
                    X_batch = X_train_shuffled[start:start+batch_size]
                    y_batch = y_train_shuffled[start:start+batch_size]
                     
                    # forward pass
                    y_pred = sf.model(X_batch)
                    loss = loss_fn(y_pred, y_batch)

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()

                    # update weights
                    optimizer.step()

                    # print progress
                    bar.set_postfix(mse=float(loss))


            # evaluate accuracy at end of each epoch

            sf.model.eval()
            y_pred = sf.model(X_test)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            history.append(mse)

            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(sf.model.state_dict())

    #   restore model and return best accuracy

        torch.save(sf.model.state_dict(), filename)
        print("MSE: %.2f" % best_mse)
        print("RMSE: %.2f" % np.sqrt(best_mse))
        plt.plot(history)
        plt.show()

    def Predict(sf, X_dat):

        X_dat = sf.scaler.transform(X_dat)
        X_dat = torch.tensor(X_dat, dtype=torch.float32)
        y_pred = sf.model(X_dat)
        y_pred = y_pred.detach().numpy()

        return(y_pred)


if __name__ == "__main__":

    X_fl = open('LOGOS-NN-Input-Training-Set.csv')
    X_fl = X_fl.readlines()
    XinpF = [ [] for i in range(len(X_fl))]

    for a,i in enumerate(X_fl):
        xalj = i.split()
        XinpF[a] = [float(xalj[j]) for j in range(0,len(xalj) ,2)]

    Yprop = open('LOGOS-NN-Prop_02_the_Training-Set.csv')
    Yprop = Yprop.read()
    Yprop = Yprop.split()
    Yprop = np.array([float(i) for i in Yprop])
    XinpF = np.array(XinpF)

    PercentSplit = 0.7
    filename = 'BestWeights_02_th.pth.tar'
    X_train, X_test, Y_train, Y_test = train_test_split(XinpF, Yprop, train_size=PercentSplit, shuffle=True, random_state=42)

    NNetwork = NeuralNetwork_MAIN()
    NNetwork.NeuralNetwork(X_train, Y_train, filename)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)
    Out_pred = NNetwork.Predict(X_test)

    print('Weights are Stored in : BestWeights_02_th.pth.tar')
