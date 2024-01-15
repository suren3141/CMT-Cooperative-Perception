'''
This file creates the training, validation and testing split for kitti_format and also the corresponding split to
convert from A9 dataset to the according kitti_format
'''
import argparse
import glob
import math
import os
import numpy as np
from pathlib import Path


def create_data_split(version, root_path, out_dir, split=None):
    is_nested = False
    num_dirs = None
    splits = []
    train_set = []
    val_set = []
    test_set = []
    dir_count = 0
    if split is None:
        # Apply this split globally
        split = [0.80, 0.10, 0.10]

    assert len(split) % 3 == 0, f'split length {len(split)} is not divisible by 3'

    if len(split) > 3:
        # This signals that we have specific splits for subdirectories
        is_nested = True
    else:
        print("Here")
        split = is_split_valid(split)[0]
    if is_nested:
        # parse each directory individually
        num_dirs = int(len(split) / 3)
        splits = is_split_valid(split)

    if version == 'point_cloud':
        # Config splits and dirs

        dirs = glob.glob(os.path.join(root_path, 'pcd_format/*'))
        if num_dirs is None:
            num_dirs = len(dirs)
            splits.extend([split for i in range(num_dirs)])
        for idx in range(num_dirs):
            dir = dirs[idx]
            split = splits[idx]
            sub_dirs = glob.glob(os.path.join(dir, '*'))
            if Path.is_file(Path(sub_dirs[0])):
                dir_count, train, val, test = create_set(dir_count, sub_dirs, split)
                train_set.extend(train)
                val_set.extend(val)
                test_set.extend(test)
                print(f'Converted {dir} with split[train: {split[0]}, val: {split[1]}, test: {split[2]}]')
            else:
                for sub_dir in sub_dirs:
                    files = glob.glob(os.path.join(sub_dir, '*'))
                    dir_count, train, val, test = create_set(dir_count, files, split)
                    train_set.extend(train)
                    val_set.extend(val)
                    test_set.extend(test)
                    print(f'Converted {sub_dir} with split [train: {split[0]}, val: {split[1]}, test: {split[2]}]')

    elif version == 'image':
        pass
    else:
        pass

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, mode=0o777, exist_ok=True)

    with open(f'{out_dir}/train.txt', 'w') as f:
        f.writelines(train_set)

    with open(f'{out_dir}/val.txt', 'w') as f:
        f.writelines(val_set)

    with open(f'{out_dir}/test.txt', 'w') as f:
        f.writelines(test_set)


# def dir_to_filename(dir, depth):
#    filenames = dir.split('/')[-depth:]
#    filename = '_'.join(filenames)
#    return filename[:-4] + '\n'

def is_split_valid(split):
    splits = []
    current_split = []
    for val in split:
        if val == 0:
            current_split.append(0.0)
        else:
            current_split.append(val / 100.0)
        if len(current_split) == 3:
            assert np.sum(current_split) == 1, f'This split {current_split} is not valid'
            splits.append(current_split)
            current_split = []
    return splits


def dir_to_filename(dir, dir_count):
    return str(dir_count).zfill(3) + str(dir).zfill(3) + '\n'


def create_set(dir_count, files, split):
    sub_dir_len = len(files)
    train_len = int(math.ceil(sub_dir_len * split[0]))
    val_len = int(math.ceil(sub_dir_len * split[1]))
    test_len = int(math.ceil(sub_dir_len * split[2]))
    # print(f'total: {sub_dir_len} - train_len: {train_len} - val_len {val_len} - test_len {test_len}')
    train = list(map(dir_to_filename, range(0, train_len), [dir_count] * train_len))
    val = list(map(dir_to_filename, range(train_len, train_len + val_len), [dir_count] * val_len))
    test = list(map(dir_to_filename, range(train_len + val_len, train_len + val_len + test_len), [dir_count] * test_len))
    return dir_count + 1, train, val, test

# Specify the version ['point_cloud', 'image', 'multi_modal']

# Specify the percentage split for each directory

# Parse over the directories

# save to list and write to file


parser = argparse.ArgumentParser(description='creating data split for A9 dataset')
parser.add_argument(
    'version',
    metavar='point_cloud',
    type=str,
    choices=['point_cloud', 'image', 'multi_modal'],
    help="Specify the version ['point_cloud', 'image', 'multi_modal']")
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/a9',
    help='Specify the root path of the dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/a9/kitti_format/ImageSets',
    help='Specify the save directory')
parser.add_argument(
    '--split',
    nargs='+',
    type=int
)
args = parser.parse_args()
if __name__ == '__main__':
    create_data_split(
        version=args.version,
        root_path=args.root_path,
        out_dir=args.out_dir,
        split=args.split)
