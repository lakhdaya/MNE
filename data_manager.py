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
from utils import create_montage
from typing import Optional, Dict, List, Tuple
from preprocessing import preprocessing_raw

class EEGDataset(Dataset):
    def __init__(
        self,
        edf_path: str,
        event_id: Dict[str, int],
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
        - edf_path: Path of the edf files
        - event_id: Dictionary mapping event labels to integers
        - tmin, tmax: Epoch time window (in seconds)
        - l_freq, h_freq: Bandpass filter frequencies
        - bad_channels: List of channel names to mark as bad
        - reject_criteria: Dictionary of rejection criteria (e.g. {'eeg': 150e-6})
        """
        self.edf_path = edf_path
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.bad_channels = bad_channels or []
        self.reject_criteria = reject_criteria
        
        self.raw = self.load_raw()
        self.event, self.event_id = mne.events_from_annotations(self.raw)
        self.events = mne.find_events(self.raw)
        self.epochs = self.make_epochs()

    def load_raw(self) -> mne.io.BaseRaw:
        """
        Load raw files from the path
        """
        raws = [preprocessing_raw(mne.io.read_raw_edf(file.path)) for file in os.scandir(self.path) if "event" not in file.name]
        return mne.concatenate_raws(raws)

    def make_epochs(self) -> mne.Epochs:
        return mne.Epochs(
            self.raw,
            self.events,
            self.events_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=(None, 0),
            reject=self.reject_criteria,
            preload=True
        )

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data: np.ndarray = self.epochs.get_data()[idx]  # shape: (n_channels, n_times)
        label: int = self.epochs.events[idx, 2]

        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor


if __name__ == "__main__":
    """
    Used to test if the dataloader works properly
    """

    print("hello world")