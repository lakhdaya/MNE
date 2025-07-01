import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from mne.preprocessing import ICA
from utils import create_montage

mne.set_log_level('WARNING')  # or 'ERROR' to reduce output further

def get_bad_channels(raw):
    data = raw.get_data()
    variances = np.var(data, axis=1)
    threshold = np.mean(variances) + 3 * np.std(variances)
    bad_idx = np.where(variances > threshold)[0]
    return [raw.ch_names[i] for i in bad_idx]

def load_subject(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)   
    raw.resample(128)
    create_montage(raw)
    raw.filter(1., 40.)
    raw.set_eeg_reference('average')
    bad_channels =  get_bad_channels(raw)
    raw.info['bads'] = bad_channels
    raw.interpolate_bads()
    # ica = ICA(n_components=20, random_state=97)
    # ica.fit(raw)
    # Basic preprocessing, adjust as needed
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=event_id, 
                        tmin=-0.2, tmax=0.8, preload=True)
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, 2]
    return data, labels

def load_dataset(folder_path):
    X_list, y_list = [], []
    for f in os.listdir(folder_path):
        if f.endswith('.edf'):
            edf_path = os.path.join(folder_path, f)
            X, y = load_subject(edf_path)
            X_list.append(X)
            y_list.append(y)

    return np.concatenate(X_list), np.concatenate(y_list)

# -------- Dataset --------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, C, T)
        self.X = self.X.permute(0, 2, 1)  # (N, T, C) for LSTM
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
