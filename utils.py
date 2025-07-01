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


import os
import shutil
import random

SUBJECTS_PATH = "subjects"

def clean_data_dir(dest_dir):
    """Supprime tous les fichiers et sous-dossiers dans le dossier data s'il existe déjà."""
    if os.path.exists(dest_dir):
        for root, dirs, files in os.walk(dest_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                shutil.rmtree(os.path.join(root, name))

def get_subjects(source_dir):
    """Returns a list of all subject folder paths inside the source_dir."""
    return [os.path.join(source_dir, d) for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))]

def split_subjects(subject_paths, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(subject_paths)
    total = len(subject_paths)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    return {
        'train': subject_paths[:train_end],
        'val': subject_paths[train_end:val_end],
        'test': subject_paths[val_end:]
    }


def create_split_dirs(dest_dir):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

def split_subjects(subject_paths, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(subject_paths)
    total = len(subject_paths)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    return {
        'train': subject_paths[:train_end],
        'val': subject_paths[train_end:val_end],
        'test': subject_paths[val_end:]
    }



def copy_subject_files(splits, dest_dir):
    for split_name, subject_paths in splits.items():
        for subject_path in subject_paths:
            for file in os.listdir(subject_path):
                if file.endswith('.edf') or file.endswith('.edf.event'):
                    src = os.path.join(subject_path, file)
                    dst = os.path.join(dest_dir, split_name, f"{os.path.basename(subject_path)}_{file}")
                    shutil.copy(src, dst)

def organize_dataset_by_subject(source_dir='subjects', dest_dir='data'):
    clean_data_dir(dest_dir)
    create_split_dirs(dest_dir)
    subjects = get_subjects(source_dir)
    splits = split_subjects(subjects)
    copy_subject_files(splits, dest_dir)
    print("Subjects organized into train, val, and test sets.")
