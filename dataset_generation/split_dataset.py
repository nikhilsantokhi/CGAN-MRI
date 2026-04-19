import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

RAW_ROOT = 'C:/Python/Python Projects/CGAN_MRI/datasets/OCMR/'
OUT_ROOT = 'C:/Python/Python Projects/CGAN_MRI/datasets/OCMR/'

R_VALUES = ['R=2', 'R=4', 'R=8']
RANDOM_STATE = 42

# prepare ground truth images
gt_ref_dir = os.path.join(RAW_ROOT, 'OCMR', 'GT')
gt_files = sorted(glob(os.path.join(gt_ref_dir, '*.png')))
assert len(gt_files) == 3302, f'Expected 3302 GT files, got {len(gt_files)}'

gt_names = [os.path.basename(f) for f in gt_files]

# split into train (80%), val (10%), test (10%)
train_val, test = train_test_split(
    gt_names, test_size=0.10, random_state=RANDOM_STATE
)
train, val = train_test_split(
    train_val, test_size=0.1111, random_state=RANDOM_STATE
)

splits = {'train': train, 'val': val, 'test': test}

# create output directories
for split in splits:
    os.makedirs(os.path.join(OUT_ROOT, 'GT', split), exist_ok=True)
    for r in R_VALUES:
        os.makedirs(os.path.join(OUT_ROOT, 'AL', r, split), exist_ok=True)

# create ground truth directory
for split, names in splits.items():
    for name in names:
        src = os.path.join(gt_ref_dir, name)
        dst = os.path.join(OUT_ROOT, 'GT', split, name)
        shutil.copy(src, dst)

# create AL directories using the same splits
for r in R_VALUES:
    al_src_dir = os.path.join(RAW_ROOT, r, 'AL')

    for split, names in splits.items():
        for gt_name in names:
            al_name = gt_name.replace('_GT_', '_AL_')

            src = os.path.join(al_src_dir, al_name)
            dst = os.path.join(OUT_ROOT, 'AL', r, split, al_name)

            if not os.path.exists(src):
                raise FileNotFoundError(f'Missing {src}')

            shutil.copy(src, dst)

print('Dataset split successfully created.')