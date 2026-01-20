# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:10:59 2026

@author: waboa
"""

#########  Import libraries   ###########
import sys
sys.path.append('Modules')
from transformer_architecture import Transformer
import optimize as opt

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###########  1. Import Data   #############
data_file_path = "Lucas/output_MissingSmallBlackWood.txt"  #output_MissingSmallBlackWood
columns_name = ["EMITTER","SMALL","MEDIUM","LARGE","WHITE","BLACK","GRAY",
                "PLASTIC","METAL","WOOD","END_SENSOR","ACT1","ACT2","ACT3","ACT4"]

data = pd.read_csv(data_file_path, names=columns_name)
data["EMITTER"] = data["EMITTER"].map(lambda x : str(x).lstrip("["))
data["ACT4"] = data["ACT4"].map(lambda x : str(x).rstrip("]"))
data = data.astype(float)


###########  2. Preprocess Data for training  #############
#Sliding windows
npast = 6

targets = ["ACT1","ACT2","ACT3","ACT4"]
X = np.array([data[i:i+npast] for i in range(len(data) - npast - 1)])
y = np.array([data[targets].iloc[i+npast] for i in range(len(data) - npast - 1)])

#Split in train and validation sets
ratio = 0.7
X_train, X_val = X[:int(ratio*len(X))], X[int(ratio*len(X)):]
y_train, y_val = y[:int(ratio*len(y))], y[int(ratio*len(y)):]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

batch_size = 128
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


########### 3. Define the Model  ################
num_features = data.shape[-1]
num_targets = len(targets)
d_model = 32
num_heads = 4
num_layers = 1
d_ff = 128
dropout = 0.1 

model = Transformer(num_features, num_targets, d_model, num_heads, num_layers, 
                    d_ff, npast, dropout, only_encoder=True).to(device)

############ 4. Train the Model  ###############   
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
activation = nn.Sigmoid()
early_stopper = opt.EarlyStopping(500, 0.0001,verbose=False, path="checkpoint.pt")

num_epochs = 1000 

start_time = time.time()
for epoch in range(num_epochs):
    opt.train(model=model, data_loader=train_loader, criterion=criterion, 
              optimizer=optimizer, device=device)  
    
    if epoch % 10 == 0: 
        train_loss =  opt.evaluate(model=model, data_loader=train_loader, 
                criterion=nn.BCELoss(),  activation=activation, device = device)
        train_accuracy = opt.accuracy(model=model, data_loader=train_loader, 
                                      activation=activation, device =device)
        val_loss =  opt.evaluate(model=model, data_loader=val_loader, 
                criterion=nn.BCELoss(),  activation=activation, device = device)
        val_accuracy = opt.accuracy(model=model, data_loader=val_loader, 
                                    activation=activation, device =device)
        print(f"Epoch {epoch}/{num_epochs} : train loss-acc = {train_loss:.4f}-{train_accuracy:.2f}%; "   
              f"val loss-acc = {val_loss:.4f}-{val_accuracy:.2f}%")
        
        if val_accuracy == 100.0 : break
    
    if early_stopper(val_loss, model):
        model.load_state_dict(torch.load('checkpoint.pt', weights_only=True))
        break

training_time = time.time() - start_time 

############ 5. Evaluate Performance ###############
train_loss =  opt.evaluate(model=model, data_loader=train_loader, criterion=nn.BCELoss(), 
                           activation=activation, device = device)
train_accuracy = opt.accuracy(model=model, data_loader=train_loader, 
                              activation=activation, device =device)
val_loss =  opt.evaluate(model=model, data_loader=val_loader, criterion=nn.BCELoss(),
                         activation=activation, device = device)
val_accuracy = opt.accuracy(model=model, data_loader=val_loader, 
                            activation=activation, device =device)

print(f"Loss train : {train_loss} ; vallidation : {val_loss}")
print(f"Accuracy train : {train_accuracy:.2f}% ; vallidation : {val_accuracy:.2f}%")
print(f"Training Time is {training_time:.3f} seconds")


############ 6. Evaluate Inference time
# Compute inference time for a single vector prediction over the validation set
total_time = 0
num_samples = 0
with torch.no_grad():
    for data, _ in val_loader:
        for i in range(data.size(0)):
            sample = data[i:i+1]
            start_inference_time = time.time()
            _ = model(sample.to(device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.time() - start_inference_time
            num_samples += 1
            if num_samples == 100:
                break

avg_inference_time = total_time/num_samples
print(f"Inference Time is {avg_inference_time:.2e} seconds")














