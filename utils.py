import mne
import numpy as np

def clean_raw(raw, montage_names):
    mapping = {}
    raw_ch_names = raw.info['ch_names']
    for ch in raw_ch_names:
        match = next((mch for mch in montage_names if mch.lower() == ch.lower().replace('.', '')), None)
        if match:
            mapping[ch] = match
    raw.rename_channels(mapping)
    raw.rename_channels({'T9..': 'T9', 'T10.': 'T10'})

def create_custom_montage(montage):
    pos = montage.get_positions()['ch_pos'].copy()
    T7 = pos['T7']
    T8 = pos['T8']
    
    pos['T9'] = T7 + np.array([-0.03, 0, 0])  # 3 cm à gauche sur X
    pos['T10'] = T8 + np.array([0.03, 0, 0])  # 3 cm à droite sur X

    ch_names = list(pos.keys())
    ch_pos = dict(zip(ch_names, [pos[ch] for ch in ch_names]))

    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

def create_montage(raw):
    montage = mne.channels.make_standard_montage('biosemi64')
    clean_raw(raw, montage.ch_names)
    raw.set_montage(create_custom_montage(montage))