"""
Dataset loader for the model
"""

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from mne.preprocessing import ICA
from typing import Optional, Dict, List, Tuple
from preprocessing import preprocessing_raw
import mne
from tqdm import tqdm

mne.set_log_level('WARNING')
class EEGDataset(Dataset):
    def __init__(
        self,
        path_edf: str,
        montage: str,
        tmin: float = -0.2,
        tmax: float = 0.8,
        l_freq: float = 1.0,
        h_freq: float = 40.0,
        bad_channels: Optional[List[str]] = None,
        reject_criteria: Optional[Dict[str, float]] = None
    ):
        """
        EEG dataset for PyTorch based on MNE Epochs.

        Parameters:
        - path_edf: Path of the edf files
        - montage: Montage name
        - tmin, tmax: Epoch time window (in seconds)
        - l_freq, h_freq: Bandpass filter frequencies
        - bad_channels: List of channel names to mark as bad
        - reject_criteria: Dictionary of rejection criteria (e.g. {'eeg': 150e-6})
        """
        self.path_edf = path_edf # path to all the edf files
        self.nb_files = len(os.listdir(self.path_edf)) // 2
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.bad_channels = bad_channels or []
        self.montage = mne.channels.make_standard_montage(montage)
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = tmax
        self.raw = self.load_raw() # load raws
        self.event, self.event_id = mne.events_from_annotations(self.raw) # get annotations from .event, this file is needed
        self.epochs = self.make_epochs()

    @classmethod
    def load(cls, raw_path: str, epochs_path: str):
        """
        Load raw and epochs from disk and create an EEGDataset instance.

        Parameters:
        - raw_path: path to saved raw file (e.g. 'raw.fif')
        - epochs_path: path to saved epochs file (e.g. 'epochs-epo.fif')

        Returns:
        - EEGDataset instance with loaded raw and epochs
        """
        raw = mne.io.read_raw_fif(raw_path, preload=True)
        epochs = mne.read_epochs(epochs_path, preload=True)

        obj = cls.__new__(cls)  # create instance without calling __init__
        obj.raw = raw
        obj.epochs = epochs
        obj.path_edf = None
        obj.l_freq = None
        obj.h_freq = None
        obj.bad_channels = None
        obj.reject_criteria = None
        obj.tmin = epochs.tmin
        obj.tmax = epochs.tmax
        obj.event, obj.event_id = mne.events_from_annotations(raw)
        return obj

    def load_raw(self) -> mne.io.BaseRaw:
        """
        Load raw files from the path
        """
        print("preprocessing raws from the folder")
        raws = [preprocessing_raw(
            mne.io.read_raw_edf(file.path, preload=True),
            self.montage,
            self.l_freq,
            self.h_freq,
            self.bad_channels,
            ) for file in tqdm(os.scandir(self.path_edf), total=self.nb_files) if "event" not in file.name]
        print("done")
        return mne.concatenate_raws(raws)

    def make_epochs(self) -> mne.Epochs:
        """
        Create epochs from raw using event taken from .event files
        """
        return mne.Epochs(
            self.raw,
            self.event,
            self.event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=(None, 0),
            reject=self.reject_criteria,
            preload=True
        )

    def __len__(self) -> int:
        """
        Our time series has a lenght depending of the number of epoch. 
        """
        return len(self.epochs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.epochs.get_data()[idx]  # shape: (n_channels, n_times)
        label = self.epochs.events[idx, 2]
        label-=1 # to go from 0 to 2 instead of 1 to 3
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor

    def save(self, raw_path: str, epochs_path: str):
        """
        Save the raw and epochs objects to disk.
        It is recommended to use it beforehand and load
        it with the ufnction load because instanciation
        can take a lot of times due to the arguemnt preload=True.
        Parameters:
        - raw_path: file path to save the raw data (e.g. 'raw.fif')
        - epochs_path: file path to save the epochs data (e.g. 'epochs-epo.fif')
        """
        self.raw.save(raw_path, overwrite=True)
        self.epochs.save(epochs_path, overwrite=True)


if __name__ == "__main__":
    """
    Used to test if the dataloader works properly. In advanced, I should use testunit
    """

    # Test creation
    data = EEGDataset(path_edf="subjects/S001", montage="standard_1005", bad_channels=["Fp1", "Fp2"])     
    
    # check format of X and Y
    X, Y = next(iter(data))
    print(X.shape, Y.shape, Y) # should be a tensor, a shpae empty and a number
    print(data.epochs.events.shape)  # Should be (n_epochs, 3)
    print(data.epochs.events[:5])    # Preview first 5 event rows

    # Test saving state
    data.save("raw.fif", "event.fif") # save in fif
    data = EEGDataset.load("raw.fif", "event.fif")
    X1, Y1 = next(iter(data))
    print(X.shape, Y1)