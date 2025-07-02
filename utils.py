import mne
import numpy as np

def rename_channel(ch_name):
    """
    Personalized function in order to swith channels names from edf file
    to math it with stanard_10005 file
    """
    ch_name = ch_name.strip('.')
    ch_name = ''.join(char.upper() if char.lower() in ['f', 'p', 'c', 'o', 't'] else char for char in ch_name)
    if ch_name in ['FP1', 'FPz', 'FP2']:
        return ch_name[:1] + 'p' + ch_name[2:]

    return ch_name