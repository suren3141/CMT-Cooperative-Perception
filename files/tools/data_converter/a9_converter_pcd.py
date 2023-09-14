import pickle
from glob import glob
from pypcd import pypcd
from tqdm import tqdm
import argparse
import json
import ntpath
import numpy as np
import os.path
import shutil


class A92KITTI(object):
    """A9 dataset to KITTI converter.

        This class serves as the converter to change the A9 data to KITTI
        format.
    """

    def __init__(self,
                 splits,
                 version,
                 load_dir,
                 save_dir,
                 name_format='name'):

        """
        Args:
            splits list[(str)]: Contains the different splits
            version (str): Specify the modality
            load_dir (str): Directory to load waymo raw data.
            save_dir (str): Directory to save data in KITTI format.
            name_format (str): Specify the output name of the converted file mmdetection3d expects names to but numbers
        """

        self.splits = splits
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.name_format = name_format
        self.label_save_dir = f'label_2'
        self.point_cloud_save_dir = f'velodyne'

        self.is_pcd = False
        self.is_image = False
        if version == 'point_cloud':
            self.is_pcd = True
        elif version == 'image':
            self.is_image = True
        else:
            self.is_pcd = True
            self.is_image = True

        self.train_set = []
        self.val_set = []
        self.test_set = []

        self.map_set_to_dir_idx = {
            'training': 0,
            'validation': 1,
            'testing': 2
        }

        self.map_version_to_dir = {
            'training': 'train',
            'validation': 'val',
            'testing': 'test'
        }

        self.imagesets = {
            'training': self.train_set,
            'validation': self.val_set,
            'testing': self.test_set
        }

        self.occlusion_map = {
            'NOT_OCCLUDED': 0,
            'PARTIALLY_OCCLUDED': 1,
            'MOSTLY_OCCLUDED': 2
        }

        self.pickle = []

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        for split in self.splits:
            split_pickle = []
            split_path = 'testing' if split == 'testing' else 'training'
            self.create_folder(split)
            print(f'Converting split: {split}...')
            #if split == 'training' or split == 'validation':
            #    continue
            file_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'point_clouds', '*')))
            for file_idx, file in enumerate(tqdm(file_list)):
                pickle = dict()
                out_filename = self.name_formatting(split, file_idx, file)
                self.save_lidar(file, os.path.join(self.point_cloud_save_dir, out_filename))
                pickle['point_cloud'] = {
                    'num_features': 4,
                    'velodyne_path': f'{split_path}/velodyne/{out_filename}.bin'
                }
                label = file.replace('point_clouds', 'labels').replace('pcd', 'json')
                label_dict = self.save_label(label, os.path.join(self.label_save_dir, out_filename))
                pickle['annos'] = label_dict
                self.imagesets[split].append(out_filename)
                self.pickle.append(pickle)
                split_pickle.append(pickle)
            self.create_pickle(os.path.join(self.save_dir, f'a9_dbinfos_{self.map_version_to_dir[split]}.pkl'), split_pickle)
        print('Creating ImageSets...')
        self.create_imagesets()
        print('Creating pickle file...')
        self.create_pickle(os.path.join(self.save_dir, 'a9_dbinfos.pkl'))
        print('\nFinished ...')

    def name_formatting(self, split, file_idx, file):
        """
        Create the specified name convention
        Args:
            split: Which split the file belongs to
            file_idx: Index of the file in the given split
            file: Filepath

        Returns: (str) Specified name

        """
        if self.name_format == 'name':
            return ntpath.basename(file).split('.')[0]
        else:
            return f'{str(self.map_set_to_dir_idx[split])}{str(file_idx).zfill(5)}'

    @staticmethod
    def save_lidar(file, out_file):
        """
        Converts file from .pcd to .bin
        Args:
            file: Filepath to .pcd
            out_file: Filepath of .bin
        """
        point_cloud = pypcd.PointCloud.from_path(file)
        np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32) / 256
        bin_format = np.column_stack((np_x, np_y, np_z, np_i))
        bin_format.tofile(os.path.join(f'{out_file}.bin'))

    def save_label(self, file, out_file):
        """
        Converts OpenLABEL format to KITTI label format
        Args:
            file: Path to label .json
            out_file: Path to .txt
        """
        # read json file
        lines = []
        json_file = open(file)
        json_data = json.load(json_file)
        labels = []
        truncs = []
        occludeds = []
        alphas = []
        bboxes = []
        dims = []
        locations = []
        rotations = []
        for id, label in json_data['frames']['objects'].items():
            category = label['object_data']['type']
            truncated = 0
            alpha = 0
            occluded = 3
            for item in label['object_data']['cuboid']['attributes']['text']:
                if item['name'] == 'occlusion_level':
                    occluded = self.occlusion_map[item['val']]
            cuboid = label['object_data']['cuboid']['val']
            x_center = cuboid[0]
            y_center = cuboid[1]
            z_center = cuboid[2]
            length = cuboid[7]
            width = cuboid[8]
            height = cuboid[9]
            _, _, yaw = self.quaternion_to_euler(cuboid[3], cuboid[4], cuboid[5], cuboid[6])
            bounding_box = [
                x_center - length / 2,
                y_center - width / 2,
                x_center + length / 2,
                y_center + width / 2,
            ]
            labels.append(category)
            truncs.append(truncated)
            occludeds.append(occluded)
            alphas.append(alpha)
            bboxes.append(bounding_box)
            dims.append([height, width, length])
            locations.append([x_center, y_center, z_center])
            rotations.append(yaw)
            line = f"{category} {round(truncated, 2)} {occluded} {round(alpha, 2)} " + \
                   f"{round(bounding_box[0], 2)} {round(bounding_box[1], 2)} {round(bounding_box[2], 2)} " + \
                   f"{round(bounding_box[3], 2)} {round(height, 2)} {round(width, 2)} {round(length, 2)} " + \
                   f"{round(x_center, 2)} {round(y_center, 2)} {round(z_center, 2)} {round(yaw, 2)}\n"
            lines.append(line)
        fp_label = open(f'{out_file}.txt', 'a')
        fp_label.writelines(lines)
        fp_label.close()
        return {
            'name': labels,
            'truncated': truncs,
            'occluded': occludeds,
            'alpha': alphas,
            'bbox': bboxes,
            'dimensions': dims,
            'location': locations,
            'rotation_y': rotations
        }

    @staticmethod
    def quaternion_to_euler(q0, q1, q2, q3):
        """
        Converts quaternions to euler angles using unique transformation via atan2

        Returns: roll, pitch and yaw

        """
        roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
        pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
        return roll, pitch, yaw

    @staticmethod
    def save_img(file, out_file):
        """
        Copies images to new location
        Args:
            file: Path to image
            out_file: Path to new location
        """
        img_path = f'{out_file}.png'
        shutil.copyfile(file, img_path)

    def create_imagesets(self):
        """
        Creates the ImageSets train.txt, val.txt, trainval.txt and test.txt each containing corresponding files
        """
        os.makedirs(os.path.join(self.save_dir, 'ImageSets'), exist_ok=True, mode=0o777)

        with open(os.path.join(self.save_dir, 'ImageSets', 'train.txt'), 'w') as file:
            file.writelines(self.train_set)

        with open(os.path.join(self.save_dir, 'ImageSets', 'val.txt'), 'w') as file:
            file.writelines(self.val_set)

        with open(os.path.join(self.save_dir, 'ImageSets', 'trainval.txt'), 'w') as file:
            file.writelines(self.train_set)
            file.writelines(self.val_set)

        with open(os.path.join(self.save_dir, 'ImageSets', 'test.txt'), 'w') as file:
            file.writelines(self.test_set)

    def create_pickle(self, fileout, file=None):
        if file is not None:
            with open(fileout, 'wb') as f:
                pickle.dump(file, f)
        else:
            with open(fileout, 'wb') as f:
                pickle.dump(self.pickle, f)

    def create_folder(self, split):
        """
        Create folder for data preprocessing.
        """
        split_path = 'testing' if split == 'testing' else 'training'
        print(split_path)
        if split != 'testing':
            dir_list1 = [
                f'velodyne'
            ]
            dir_list2 = [f'label_2']
        else:
            dir_list1 = [
                f'velodyne'
            ]
            dir_list2 = []
        for d in dir_list1:
            self.point_cloud_save_dir = os.path.join(self.save_dir, split_path, d)
            print(self.point_cloud_save_dir)
            os.makedirs(self.point_cloud_save_dir, exist_ok=True, mode=0o777)
        for d in dir_list2:
            self.label_save_dir = os.path.join(self.save_dir, split_path, d)
            os.makedirs(self.label_save_dir, exist_ok=True, mode=0o777)
