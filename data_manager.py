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
