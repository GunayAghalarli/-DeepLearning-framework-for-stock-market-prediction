#!/usr/bin/env python
# coding: utf-8

# In[22]:


# get_ipython().system('pip3 install PyWavelets')
# get_ipython().system('pip3 install statsmodels')
import pywt
from statsmodels.robust import mad
import numpy as np
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import sklearn
import time
import os
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[23]:

# this project code is written taking help from https://github.com/mlpanda/DeepLearning_Financial

# This function is use to denoise the input using wavelet approach
def waveletSmooth(x, wavelet="db4", level=1, DecLvl=2, title=None):
    """
       x is an individual column of the dataset
       """
    # here we are calculating wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per", level=DecLvl)
    # calculate a threshold
    sigma = mad(coeff[-level])
    # apply wavelet denoising
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    y = pywt.waverec(coeff, wavelet, mode="per")
    return y


# In[24]:


class Autoencoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden=10, sparsity_target=0.05, sparsity_weight=0.2, lr=0.01,
                 weight_decay=0.0):  # lr=0.0001):
        super(Autoencoder, self).__init__()
        # shape of input
        self.n_in = n_in
        # number of hidden units
        self.n_hidden = n_hidden
        # sparsity target
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        # decay weight
        self.weight_decay = weight_decay
        self.lr = lr
        self.build_model()

    def build_model(self):
        # encoder model
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_hidden//2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.n_hidden//2, self.n_hidden // 2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.n_hidden // 2, self.n_hidden),
            torch.nn.Sigmoid()
        )
        # decoder model
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.n_in))  # ,
        # torch.nn.Sigmoid())
        # loss function from torch basically L1 loss
        self.l1_loss = torch.nn.L1Loss(size_average=False)
        # adam optimiser
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay)

    def forward(self, inputs):
        # forward network will take encoder first
        hidden = self.encoder(inputs)
        hidden_mean = torch.mean(hidden, dim=0) # output of autoencoder
        sparsity_loss = torch.sum(self.kl_divergence(self.sparsity_target, hidden_mean))
        # decoder will take the final hidden layer from encoder
        return self.decoder(hidden), sparsity_loss

    # taken from paper
    def kl_divergence(self, p, q):
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))  # Kullback Leibler divergence

    def fit(self, X, n_epoch=10, batch_size=64, en_shuffle=True):
        for epoch in range(n_epoch):
            # if the data has to be shuffled
            if en_shuffle:
                X = sklearn.utils.shuffle(X)
            # training ...
            for local_step, X_batch in enumerate(self.gen_batch(X, batch_size)):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
                # forward pass
                outputs, sparsity_loss = self.forward(inputs)

                # calculate loss
                l1_loss = self.l1_loss(outputs, inputs)
                loss = l1_loss + self.sparsity_weight * sparsity_loss
                print(f"Autoencoder ======== epoch {epoch} / {n_epoch}, Loss: {loss}")
                self.optimizer.zero_grad()  # clear gradients for this training step
                # backward prop
                loss.backward()  # backpropagation, compute gradients
                # do weight update
                self.optimizer.step()  # apply gradients

                # if local_step % 50 == 0:
                #     print ("Epoch %d/%d | Step %d/%d | train loss: %.4f | l1 loss: %.4f | sparsity loss: %.4f"
                #            %(epoch+1, n_epoch, local_step, len(X)//batch_size,
                #              loss.data[0], l1_loss.data[0], sparsity_loss.data[0]))

    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i: i + batch_size]


# # In[25]:


class Sequence(nn.Module):
    def __init__(self, nb_features=1, hidden_size=100, nb_layers=5, dropout=0.5):
        super(Sequence, self).__init__()
        self.nb_features = nb_features
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers
        self.lstm = nn.LSTM(self.nb_features, self.hidden_size, self.nb_layers, dropout=dropout)
        self.lin = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        # print(type(h0))
        c0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        # print(type(c0))
        output, hn = self.lstm(input, (h0, c0))
        # output = F.relu(self.lin(output))
        out = self.lin(output[-1])
        return out


# In[26]:


def prepare_data_lstm(x_encoded, y_close, time_steps, log_return=True, train=True):
    ct = 0
    data = []
    for i in range(len(x_encoded) - time_steps):
        ct += 1
        if train:
            x_train = x_encoded[i:i + time_steps]
        else:
            x_train = x_encoded[:i + time_steps]

        data.append(x_train)

    if log_return == False:
        y_close = y_close.pct_change()[1:]
    else:
        y_close = (np.log(y_close) - np.log(y_close.shift(1)))[1:]  # the log return, i.e. ln(y_t/y_(t-1))

    if train:
        y = y_close[time_steps - 1:]
    else:
        y = y_close

    return data, y


class ExampleDataset(Dataset):

    def __init__(self, x, y, batchsize):
        self.datalist = x
        self.target = y
        self.batchsize = batchsize
        self.length = 0
        self.length = len(x)

    def __len__(self):
        return int(self.length / self.batchsize + 1)

    def __getitem__(self, idx):
        x = self.datalist[idx * self.batchsize:(idx + 1) * self.batchsize]
        y = self.target[idx * self.batchsize:(idx + 1) * self.batchsize]
        sample = {'x': x, 'y': y}

        return sample


def evaluate_lstm(dataloader, model, criterion):
    pred_val = []
    target_val = []
    model.eval()
    # do evaluation
    loss_val = 0
    sample_cum_x = [None]

    for j in range(len(dataloader)):

        sample = dataloader[j]
        sample_x = sample["x"]

        if len(sample_x) != 0:
            sample_x = np.stack(sample_x)
            input = Variable(torch.FloatTensor(sample_x), requires_grad=False)
            input = torch.transpose(input, 0, 1)
            target = Variable(torch.FloatTensor(sample["y"].values), requires_grad=False)

            out = model(input)

            loss = criterion(out, target)

            loss_val += float(loss.data.numpy())
            pred_val.extend(out.data.numpy().flatten().tolist())
            target_val.extend(target.data.numpy().flatten().tolist())

    return loss_val, pred_val, target_val


def backtest(predictions, y):
    trans_cost = 0.0001
    real = [1]
    index = [1]
    profit_buy, profit_sell = [], []
    for r in range(len(predictions)):
        rets = y.values.flatten().tolist()
        ret = rets[r]
        real.append(real[-1] * (1 + ret))

        if predictions[r] > 0.0:
            # buy
            ret = rets[r] - 2 * trans_cost
            index.append(index[-1] * (1 + ret))
            profit_buy.append(predictions[r] / index[-1])

        elif predictions[r] < 0.0:
            # sell
            ret = -rets[r] - 2 * trans_cost
            index.append(index[-1] * (1 + ret))
            profit_sell.append(predictions[r] / index[-1])
        else:
            # print("no trade")
            # don't trade
            index.append(index[-1])
    R = 100 * (sum(profit_buy) + sum(profit_sell))
    mape = 0
    for predictions, actual in zip(index, real):
        mape += (predictions - actual) / actual
    mape = mape / len(index)
    print("MAPE: ", mape)
    return index, real, R


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', name="checkpoint"):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (name) + 'model_best.pth.tar')


# In[27]:


# ---------------------------------------------------------------------------
# --------------------------- STEP 0: LOAD DATA -----------------------------
# ---------------------------------------------------------------------------

path = "./dataset/US1.AAPL_140623_150623.csv"
data_master = pd.read_csv(path, sep=",")

# 600 is a bit more than 2 years of data
num_datapoints = 6770
# roll by approx. 60 days - 3 months of trading days
step_size = int(0.1 * num_datapoints)
# calculate number of iterations we can do over the entire data set
num_iterations = int(np.ceil((len(data_master) - num_datapoints) / step_size)) + 2

y_test_lst = []
preds = []
ct = 0

for n in range(num_iterations):
    print(n)
    data = data_master.iloc[n * step_size:num_datapoints + n * step_size, :]
    data.columns = [col.strip() for col in data.columns.tolist()]
    print(data.shape)
    ct += 1

    feats = data.iloc[:, 2:]

    # # This is a scaling of the inputs such that they are in an appropriate range
    feats["<CLOSE>"].loc[:] = feats["<CLOSE>"].loc[:] / 1000
    feats["<OPEN>"].loc[:] = feats["<OPEN>"].loc[:] / 1000
    feats["<HIGH>"].loc[:] = feats["<HIGH>"].loc[:] / 1000
    feats["<LOW>"].loc[:] = feats["<LOW>"].loc[:] / 1000
    feats["<VOL>"].loc[:] = feats["<VOL>"].loc[:] / 1000000

    data_close = feats["<CLOSE>"].copy()
    data_close_new = data_close

    # Split in train, test and validation set

    test = feats[-step_size:]
    validate = feats[-2 * step_size:-step_size]
    train = feats[:-2 * step_size]

    y_test = data_close_new[-step_size:].values
    y_validate = data_close_new[-2 * step_size:-step_size].values
    y_train = data_close_new[:-2 * step_size].values
    feats_train = train.values.astype(np.float)
    feats_validate = validate.values.astype(np.float)
    feats_test = test.values.astype(np.float)

    # ---------------------------------------------------------------------------
    # ----------------------- STEP 2.0: NORMALIZE DATA --------------------------
    # ---------------------------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(feats_train)
    feats_train = scaler.transform(feats_train)
    feats_validate = scaler.transform(feats_validate)
    feats_test = scaler.transform(feats_test)


    data_close = pd.Series(np.concatenate((y_train, y_validate, y_test)))

    feats_norm_train = feats_train.copy()
    feats_norm_validate = feats_validate.copy()
    feats_norm_test = feats_test.copy()

    # ---------------------------------------------------------------------------
    # ----------------------- STEP 2.1: DENOISE USING DWT -----------------------
    # ---------------------------------------------------------------------------

    for i in range(feats_norm_train.shape[1]):
        feats_norm_train[:, i] = waveletSmooth(feats_norm_train[:, i], level=1)[-len(feats_norm_train):]

    # for the validation we have to do the transform using training data + the current and past validation data
    # i.e. we CAN'T USE all the validation data because we would then look into the future
    temp = np.copy(feats_norm_train)
    feats_norm_validate_WT = np.copy(feats_norm_validate)
    for j in range(feats_norm_validate.shape[0]):
        # first concatenate train with the latest validation sample
        temp = np.append(temp, np.expand_dims(feats_norm_validate[j, :], axis=0), axis=0)
        for i in range(feats_norm_validate.shape[1]):
            feats_norm_validate_WT[j, i] = waveletSmooth(temp[:, i], level=1)[-1]

    # for the test we have to do the transform using training data + validation data + current and past test data
    # i.e. we CAN'T USE all the test data because we would then look into the future
    temp_train = np.copy(feats_norm_train)
    temp_val = np.copy(feats_norm_validate)
    temp = np.concatenate((temp_train, temp_val))
    feats_norm_test_WT = np.copy(feats_norm_test)
    for j in range(feats_norm_test.shape[0]):
        # first concatenate train with the latest validation sample
        temp = np.append(temp, np.expand_dims(feats_norm_test[j, :], axis=0), axis=0)
        for i in range(feats_norm_test.shape[1]):
            feats_norm_test_WT[j, i] = waveletSmooth(temp[:, i], level=1)[-1]

    # ---------------------------------------------------------------------------
    # ------------- STEP 3: ENCODE FEATURES USING AUTOENCODER -----------
    # ---------------------------------------------------------------------------




    n_epoch = 50  # 20000
    start_time = time.time()
    # ---- train using training data

    # The n==0 statement is done because we only want to initialize the network once and then keep training
    # as we move through time

    # if n == 0:
    #     auto1 = Autoencoder(feats_norm_train.shape[1], num_hidden_1)
    # auto1.fit(feats_norm_train, n_epoch=n_epoch)
    #
    # inputs = torch.autograd.Variable(torch.from_numpy(feats_norm_train.astype(np.float32)))
    #
    # if n == 0:
    #     auto2 = Autoencoder(num_hidden_1, num_hidden_2)
    # auto1_out = auto1.encoder(inputs).data.numpy()
    # auto2.fit(auto1_out, n_epoch=n_epoch)
    #
    # auto1_out = torch.autograd.Variable(torch.from_numpy(auto1_out.astype(np.float32)))
    #
    # if n == 0:
    #     auto3 = Autoencoder(num_hidden_2, num_hidden_3)
    # auto2_out = auto2.encoder(auto1_out).data.numpy()
    # auto3.fit(auto2_out, n_epoch=n_epoch)
    #
    # auto2_out = torch.autograd.Variable(torch.from_numpy(auto2_out.astype(np.float32)))
    #
    # if n == 0:
    #     auto4 = Autoencoder(num_hidden_3, num_hidden_4)
    # auto3_out = auto3.encoder(auto2_out).data.numpy()
    # auto4.fit(auto3_out, n_epoch=n_epoch)
    #
    # print("Total training time: ", time.time() - start_time)
    # # Change to evaluation mode, in this mode the network behaves differently, e.g. dropout is switched off and so on
    # auto1.eval()
    # auto2.eval()
    # auto3.eval()
    # auto4.eval()


    train_encoded = feats_norm_train

    validate_encoded = feats_norm_validate_WT

    test_encoded = feats_norm_test_WT


    # switch back

    # ---------------------------------------------------------------------------
    # -------------------- STEP 4: PREPARE TIME-SERIES --------------------------
    # ---------------------------------------------------------------------------

    # split the entire training time-series into pieces, depending on the number
    # of time steps for the LSTM

    time_steps = 4

    args = (train_encoded, validate_encoded, test_encoded)

    x_concat = np.concatenate(args)

    validate_encoded_extra = np.concatenate((train_encoded[-time_steps:], validate_encoded))
    test_encoded_extra = np.concatenate((validate_encoded[-time_steps:], test_encoded))

    y_train_input = data_close[:-len(validate_encoded) - len(test_encoded)]
    y_val_input = data_close[-len(test_encoded) - len(validate_encoded) - 1:-len(test_encoded)]
    y_test_input = data_close[-len(test_encoded) - 1:]

    x, y = prepare_data_lstm(train_encoded, y_train_input, time_steps, log_return=True, train=True)
    x_v, y_v = prepare_data_lstm(validate_encoded_extra, y_val_input, time_steps, log_return=False, train=False)
    x_te, y_te = prepare_data_lstm(test_encoded_extra, y_test_input, time_steps, log_return=False, train=False)

    x_test = x_te
    x_validate = x_v
    x_train = x

    y_test = y_te
    y_validate = y_v
    y_train = y

    y_train = y_train.values

    # ---------------------------------------------------------------------------
    # ------------- STEP 5: TIME-SERIES REGRESSION USING LSTM -------------------
    # ---------------------------------------------------------------------------

    batchsize = 10

    trainloader = ExampleDataset(x_train, y_train, batchsize)
    valloader = ExampleDataset(x_validate, y_validate, 1)
    testloader = ExampleDataset(x_test, y_test, 1)

    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # build the model
    if n == 0:
        seq = Sequence(7, hidden_size=64, nb_layers=4)

    resume = ""

    # if a path is given in resume, we resume from a checkpoint
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        seq.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in seq.parameters()])))

    # we use the mean squared error loss
    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=seq.parameters(), lr=0.0005)

    start_epoch = 0
    epochs = 25 # 5000

    global_loss_val = np.inf
    # begin to train
    global_profit_val = -np.inf

    for i in range(start_epoch, epochs):
        seq.train()
        loss_train = 0

        # shuffle ONLY training set
        combined = list(zip(x_train, y_train))
        random.shuffle(combined)
        x_train = []
        y_train = []
        x_train[:], y_train[:] = zip(*combined)

        # initialize trainloader with newly shuffled training data
        trainloader = ExampleDataset(x_train, y_train, batchsize)

        pred_train = []
        target_train = []
        for j in range(len(trainloader)):
            sample = trainloader[j]
            sample_x = sample["x"]

            if len(sample_x) != 0:
                sample_x = np.stack(sample_x)
                input = Variable(torch.FloatTensor(sample_x), requires_grad=False)
                input = torch.transpose(input, 0, 1)
                target = Variable(torch.FloatTensor([x for x in sample["y"]]), requires_grad=False)

                optimizer.zero_grad()
                out = seq(input)
                loss = criterion(out, target)
                print("RMSE loss: ", loss.item())
                loss_train += float(loss.data.numpy())
                pred_train.extend(out.data.numpy().flatten().tolist())
                target_train.extend(target.data.numpy().flatten().tolist())

                loss.backward()

                optimizer.step()

        if i % 50 == 0:

            plt.plot(pred_train, label = "Predicted train")
            plt.plot(target_train, label = "Actual train")
            plt.legend()
            # plt.show()

            loss_val, pred_val, target_val = evaluate_lstm(dataloader=valloader, model=seq, criterion=criterion)
            print("RMSE loss on validation data: ", loss_val)
            plt.plot(pred_val,label="Predicted validation")
            plt.plot(target_val, label=" Actual validation")
            # plt.scatter(range(len(pred_val)), pred_val, label="Predictions on validation")
            # plt.scatter(range(len(pred_val)), target_val, label=" Actual validation")
            plt.legend()
            # plt.show()

            index, real, R = backtest(pred_val, y_validate)
            print("Strategy profitabliity for validation data: ", R)

            print("Effective real cost of today: ", index[-1])
            print("Prediction: ", pred_val[-1])
            # save according to profitability
            if index[-1] > global_profit_val and i > 200:
                print("CURRENT BEST")
                global_profit_val = index[-1]
                save_checkpoint({'epoch': i + 1, 'state_dict': seq.state_dict()}, is_best=True,
                                filename='checkpoint_lstm.pth.tar')

            save_checkpoint({'epoch': i + 1, 'state_dict': seq.state_dict()}, is_best=False,
                            filename='checkpoint_lstm.pth.tar')

            print("LOSS TRAIN: " + str(float(loss_train)))
            print("LOSS VAL: " + str(float(loss_val)))
            print(i)

    # do the final test
    # first load the best checkpoint on the val set

    resume = "./runs/checkpoint/model_best.pth.tar"
    # resume = "./runs/HF/checkpoint_lstm.pth.tar"

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        seq.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    seq.eval()

    loss_test, preds_test, target_test = evaluate_lstm(dataloader=testloader, model=seq, criterion=criterion)

    print("LOSS TEST: " + str(float(loss_test)))

    temp2 = y_test.values.flatten().tolist()
    y_test_lst.extend(temp2)

    plt.plot(preds_test, label= "Predicted test data")
    plt.plot(y_test_lst, label= "Actual test data")
    # plt.scatter(range(len(preds_test)), preds_test)
    # plt.scatter(range(len(y_test_lst)), y_test_lst)
    plt.legend()
    # plt.show()
    plt.savefig("test_preds.pdf")

    # ---------------------------------------------------------------------------
    # ------------------ STEP 6: BACKTEST (ARTICLE WAY) -------------------------
    # ---------------------------------------------------------------------------

    index, real, R = backtest(preds_test, pd.DataFrame(y_test_lst))
    print("Strategy profitabliity: ", R)
    plt.close()
    plt.plot(index, label="strat")
    plt.plot(real, label="bm")
    plt.legend()
    plt.savefig("performance_article_way.pdf")
    # plt.show()
    plt.close()

# In[ ]:


# In[ ]:




