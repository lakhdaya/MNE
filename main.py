"""
Main component, add argparse later
MAIN TODO NOT functional right now
"""

import torch
from data_manager import EEGDataset
from torch.utils.data import Dataset, DataLoader
from model import EEG_LSTM
import torch.nn as nn
import numpy as np
from train import train_epochs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 5

    print("Preprocessing data")
    #load data, using standard_1005 such as precized in the link
    train_dataset = EEGDataset("data/val","standard_1005") # using val for now because fucntions too slow
    valid_dataset = EEGDataset("data/val", "standard_1005") #

    #load loader, in windows, we can't use multiple workers and GPU
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0) # dataset is small memory so we can use big batch_size
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=0)

    sample, _ = train_dataset[0]
    n_channels = sample.shape[1] # get number of entry channel, usually 64 for our case
    num_classes = 3 #fixed for T0, T1 and T2

    #actual training is just a test not the optimized one
    model = EEG_LSTM(input_size=n_channels, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    print(next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss() # default loss to test, to refined after
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)# default optimized, to refined 
    print("Traning phase")
    print(f'Using device: {device}')
    train_epochs(
        model, 
        train_loader, 
        valid_loader, 
        criterion, 
        optimizer, 
        device, 
        epochs)
    

if __name__ == "__main__":
    main()