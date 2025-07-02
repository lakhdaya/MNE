
from utils import rename_channel
import mne

def preprocessing_raw(raw, montage, l_freq: float, h_freq: float, bad_channels):
    raw.rename_channels(rename_channel) # rename channels to make it work with standard1005
    raw.filter(l_freq, h_freq) # filter with low and high
    if bad_channels:
        raw.info['bads'].extend(bad_channels)
    raw.set_eeg_reference('average', projection=False)
    raw.set_montage(montage)
    if raw.info['bads']:
        raw.interpolate_bads()
    return raw