import torch
from data_manager import EEGDataset, load_dataset
from torch.utils.data import Dataset, DataLoader
from model import EEG_LSTM
import torch.nn as nn
import numpy as np
from train import train, evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    X_train, y_train = load_dataset('data/train')
    print(len(X_train))
    X_valid, y_valid = load_dataset('data/valid')

    train_dataset = EEGDataset(X_train, y_train)
    valid_dataset = EEGDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    n_channels = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = EEG_LSTM(input_size=n_channels, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        print(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Acc: {train_acc:.4f} - Valid loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}')



if __name__ == "__main__":
    main()