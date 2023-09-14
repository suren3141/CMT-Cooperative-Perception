import json
import os.path
from glob import glob
import mmcv
from pypcd import pypcd
import numpy as np
import shutil
import ntpath
from tqdm import tqdm

from scipy.spatial.transform import Rotation

class A92Nusc(object):
    """A9 dataset to KITTI converter.

        This class serves as the converter to change the A9 data to Nuscenes format.
    """

    def __init__(self,
                 splits,
                 load_dir,
                 save_dir,
                 name_format='name'):

        """
        Args:
            splits list[(str)]: Contains the different splits
            version (str): Specify the modality
            load_dir (str): Directory to load A9 raw data.
            save_dir (str): Directory to save data in Nuscenes format.
            name_format (str): Specify the output name of the converted file mmdetection3d expects names to but numbers
        """

        self.splits = splits
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.name_format = name_format
        self.label_save_dir = f'label_2'
        self.point_cloud_save_dir = f'point_clouds'

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
            split_path = self.map_version_to_dir[split]
            self.create_folder(split)
            print(f'Converting split: {split}...')
            
            # Delete when testing split in dataset no longer has broken labels
            #if split == 'testing':
            #    continue

            test = False
            if split == 'testing':
                test = True

            pcd_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'point_clouds', 's110_lidar_ouster_south', '*.pcd')))
            
            for idx, pcd in enumerate(tqdm(pcd_list)):
                out_filename = os.path.splitext(pcd)[0]
                out_filename = os.path.split(out_filename)[-1]

                out_filename = os.path.join(self.point_cloud_save_dir, out_filename)

                self.save_lidar(pcd, out_filename)
                pcd_list[idx] = out_filename+'.bin'
                  
            img_south1_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'images', 's110_camera_basler_south1_8mm', '*')))
            img_south2_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'images', 's110_camera_basler_south2_8mm', '*')))
            pcd_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_point_clouds', 's110_lidar_ouster_south', '*')))
            img_south1_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_images', 's110_camera_basler_south1_8mm', '*')))
            img_south2_labels_list = sorted(glob(os.path.join(self.load_dir, self.map_version_to_dir[split], 'labels_images', 's110_camera_basler_south2_8mm', '*')))

            # print(pcd_list)
            # print(img_south1_list)
            # print(img_south2_list)
            # print(pcd_labels_list, img_south1_labels_list, img_south2_labels_list)

            infos_list = self._fill_infos(pcd_list, img_south1_list, img_south2_list, pcd_labels_list, img_south1_labels_list, img_south2_labels_list, test)

            metadata = dict(version='r1')

            if test:
                print("test sample: {}".format(len(infos_list)))
                data = dict(infos=infos_list, metadata=metadata)
                info_path = os.path.join(self.save_dir, "{}_infos_test.pkl".format('a9_nusc'))
                mmcv.dump(data, info_path)
            else:
                if split == 'training':
                    print("train sample: {}".format(len(infos_list)))
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, "{}_infos_train.pkl".format('a9_nusc'))
                    mmcv.dump(data, info_path)
                elif split == 'validation':
                    print("val sample: {}".format(len(infos_list)))
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, "{}_infos_val.pkl".format('a9_nusc'))
                    mmcv.dump(data, info_path)

        print('\nFinished ...')

    def _fill_infos(self, pcd_list, img_south1_list, img_south2_list, pcd_labels_list, img_south1_labels_list, img_south2_labels_list, test=False):
        infos_list = []

        lidar2ego = np.asarray([[0.99011437, -0.13753536, -0.02752358, 2.3728100375737995],
                                [0.13828977, 0.99000475, 0.02768645, -16.19297517556697],
                                [0.02344061, -0.03121898, 0.99923766, -8.620000000000005],
                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        lidar2ego = lidar2ego[:-1, :]

        lidar2s1image = np.asarray([[7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
                                    [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
                                    [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00]], dtype=np.float32)

        lidar2s2image = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                                    [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                                    [0.73326062, 0.59708904, -0.32528854, -1.30114325]], dtype=np.float32)
        
        south1intrinsics = np.asarray([[1400.3096617691212, 0.0, 967.7899705163408],
                                       [0.0, 1403.041082755918, 581.7195041357244],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        
        south12ego = np.asarray([[-0.06377762, -0.91003007, 0.15246652, -10.409943],
                                 [-0.41296193, -0.10492031, -0.8399004, -16.2729],
                                 [0.8820865, -0.11257353, -0.45447016, -11.557314],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        south12ego = south12ego[:-1, :]

        south12lidar = np.asarray([[-0.10087585, -0.51122875, 0.88484734, 1.90816304],
                                   [-1.0776537, 0.03094424, -0.10792235, -14.05913251],
                                   [0.01956882, -0.93122171, -0.45454375, 0.72290242],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        south12lidar = south12lidar[:-1, :]
        
        south2intrinsics = np.asarray([[1029.2795655594014, 0.0, 982.0311857478633],
                                       [0.0, 1122.2781391971948, 1129.1480997238505],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        
        south22ego = np.asarray([[0.650906, -0.7435749, 0.15303044, 4.6059465],
                                 [-0.14764456, -0.32172203, -0.935252, -15.00049],
                                 [0.74466264, 0.5861663, -0.3191956, -9.351643],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        south22ego = south22ego[:-1, :]

        south22lidar = np.asarray([[0.49709212, -0.19863714, 0.64202357, -0.03734614],
                                   [-0.60406415, -0.17852863, 0.50214409, 2.52095055],
                                   [0.01173726, -0.77546627, -0.70523436, 0.54322305],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        south22lidar = south22lidar[:-1, :]

        for i, pcd_path in enumerate(pcd_list):
            json1_file = open(pcd_labels_list[i])
            json1_str = json1_file.read()
            lidar_annotation = json.loads(json1_str)

            lidar_anno_frame = {}

            for j in lidar_annotation['openlabel']['frames']:
                lidar_anno_frame = lidar_annotation['openlabel']['frames'][j]

            info = {
                "lidar_path": pcd_path,
                "lidar_anno_path": pcd_labels_list[i],
                "sweeps": [],
                "cams": dict(),
                "lidar2ego": lidar2ego,
                "timestamp": lidar_anno_frame['frame_properties']['timestamp'],
                "location": lidar_anno_frame['frame_properties']['point_cloud_file_names'][0].split("_")[2],
            }

            json2_file = open(img_south1_labels_list[i])
            json2_str = json2_file.read()
            south1_annotation = json.loads(json2_str)

            south1_anno_frame = {}

            for k in south1_annotation['openlabel']['frames']:
                south1_anno_frame = south1_annotation['openlabel']['frames'][k]

            img_south1_info = {
                "data_path": img_south1_list[i],
                "type": 's110_camera_basler_south1_8mm',
                "lidar2image": lidar2s1image,
                "sensor2ego": south12ego,
                "sensor2lidar": south12lidar,
                "camera_intrinsics": south1intrinsics,
                "timestamp": south1_anno_frame['frame_properties']['timestamp'],
            }
            
            info["cams"].update({'s110_camera_basler_south1_8mm': img_south1_info})

            json3_file = open(img_south2_labels_list[i])
            json3_str = json3_file.read()
            south2_annotation = json.loads(json3_str)

            south2_anno_frame = {}

            for l in south2_annotation['openlabel']['frames']:
                south2_anno_frame = south2_annotation['openlabel']['frames'][l]

            img_south2_info = {
                "data_path": img_south2_list[i],
                "type": 's110_camera_basler_south2_8mm',
                "lidar2image": lidar2s2image,
                "sensor2ego": south22ego,
                "sensor2lidar": south22lidar,
                "camera_intrinsics": south2intrinsics,
                "timestamp": south2_anno_frame['frame_properties']['timestamp'],
            }
            
            info["cams"].update({'s110_camera_basler_south2_8mm': img_south2_info})

            # obtain annotation

            if not test:
                gt_boxes = []
                gt_names = []
                velocity = []
                valid_flag = []
                num_lidar_pts = []
                num_radar_pts = []

                for id in lidar_anno_frame['objects']:
                    object_data = lidar_anno_frame['objects'][id]['object_data']
                    
                    loc = np.asarray(object_data['cuboid']['val'][:3], dtype=np.float32)
                    dim = np.asarray(object_data['cuboid']['val'][7:], dtype=np.float32)
                    rot = np.asarray(object_data['cuboid']['val'][3:7], dtype=np.float32) # Quaternion in x,y,z,w

                    rot_temp = Rotation.from_quat(rot)
                    rot_temp = rot_temp.as_euler('xyz', degrees=False)

                    yaw = np.asarray(rot_temp[2], dtype=np.float32)

                    gt_box = np.concatenate([loc, dim, -yaw], axis=None)

                    gt_boxes.append(gt_box)
                    gt_names.append(object_data['type'])
                    velocity.append([0, 0])
                    valid_flag.append(True)

                    for n in object_data['cuboid']['attributes']['num']:
                        if n['name'] == 'num_points':
                            num_lidar_pts.append(n['val'])
                    
                    num_radar_pts.append(0)

                gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                info['gt_boxes'] = gt_boxes
                info['gt_names'] = np.array(gt_names)
                info["gt_velocity"] = np.array(velocity).reshape(-1, 2)
                info["num_lidar_pts"] = np.array(num_lidar_pts)
                info["num_radar_pts"] = np.array(num_radar_pts)
                info["valid_flag"] = np.array(valid_flag, dtype=bool)

            infos_list.append(info)
            
        return infos_list

    @staticmethod
    def save_lidar(pcd_file, out_file):
        """
        Converts file from .pcd to .bin
        Args:
            file: Filepath to .pcd
            out_file: Filepath of .bin
        """
        # print(pcd_file, out_file)
        point_cloud = pypcd.PointCloud.from_path(pcd_file)
        np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32) / 256
        np_ts = np.zeros((np_x.shape[0],), dtype=np.float32)
        bin_format = np.column_stack((np_x, np_y, np_z, np_i, np_ts)).flatten()
        bin_format.tofile(os.path.join(f'{out_file}.bin'))

    @staticmethod
    def save_img(file, out_file):
        """
        Copies images to new location
        Args:
            file: Path to image
            out_file: Path to new location
        """
        img_path = f'{out_file}.jpg'
        shutil.copyfile(file, img_path)

    def create_folder(self, split):
        """
        Create folder for data preprocessing.
        """
        split_path = self.map_version_to_dir[split]
        dir_list1 = [f'point_clouds/s110_lidar_ouster_south']
        for d in dir_list1:
            self.point_cloud_save_dir = os.path.join(self.save_dir, split_path, d)
            os.makedirs(self.point_cloud_save_dir, exist_ok=True, mode=0o777)


class A92KITTI(object):
    """A9 dataset to KITTI converter.

        This class serves as the converter to change the A9 data to KITTI
        format.

        Args:
            load_dir (str): Directory to load waymo raw data.
            save_dir (str): Directory to save data in KITTI format.
            prefix (str): Prefix of filename. In general, 0 for training, 1 for
                validation and 2 for testing.
            workers (int, optional): Number of workers for the parallel process.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 version,
                 prefix,
                 workers=0,
                 test_mode=False):

        self.selected_a9_classes = []  # TODO: Add the chosen classes
        self.a9_to_kitti_class_map = {}  # TODO: Add the mapping

        self.is_pcd = False
        self.is_img = False
        self.is_multi_modal = False

        if version == 'point_cloud':
            self.is_pcd = True
        elif version == 'image':
            self.is_img = True
        else:
            self.is_multi_modal = True

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.version = version
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.image_save_dir = f'{self.save_dir}/image_'
        self.label_save_dir = f'{self.save_dir}/label_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.dir_list = []
        self.create_folder()
        if not os.path.exists(self.point_cloud_save_dir):
            os.makedirs(self.point_cloud_save_dir)
        dir_list_pcd = glob(os.path.join(self.load_dir, 'pcd_format', '*'))
        dir_list_img = glob(os.path.join(self.load_dir, 'images', '*'))
        dir_list_calib = glob(os.path.join(self.load_dir, 'calib', '*'))
        dir_list_labels = glob(os.path.join(self.load_dir, 'labels', '*'))

        # TODO: Create fitting tuples

        # Create tuple with ("split_name", is_pcd, is_img, is_calib)
        dir_pcd_clean_list = []
        dir_img_clean_list = []
        dir_calib_clean_list = []
        dir_label_clean_list = []
        for dir_pcd in dir_list_pcd:
            dir_pcd_cleaned = dir_pcd.replace("_point_clouds", "")
            dir_pcd_clean_list.append(dir_pcd_cleaned)

        for dir_img in dir_list_img:
            dir_img_cleaned = dir_img.replace("_images", "")
            dir_img_clean_list.append(dir_img_cleaned)

        for dir_label in dir_list_labels:
            dir_label_cleaned = dir_label.replace("_labels", "")
            dir_label_clean_list.append(dir_label_cleaned)

        # Dependant on version chosen
        process_list_dir = []
        process_list_label_dir = []
        if self.version == 'point_cloud':
            process_list_dir.extend(dir_pcd_clean_list)
            process_list_label_dir.extend([label for label in dir_label_clean_list if 'point_cloud' in label])
        elif self.version == 'image':
            process_list_dir.extend(dir_img_clean_list)
            process_list_label_dir.extend([label for label in dir_label_clean_list if 'point_cloud' not in label])
        else:
            # TODO: Multi Modal
            pass

        # Get matching here
        process_list_dir = self.match(process_list_dir, process_list_label_dir)
        for process_dir in process_list_dir:
            process_target, process_dir_label = process_dir
            check_dir_list = glob(os.path.join(process_target + "_" + self.version + "s", '*'))
            check_dir_label_list = glob(os.path.join(process_dir_label + "_labels", '*'))
            for target_dir, label_dir in zip(check_dir_list, check_dir_label_list):
                if os.path.isdir(target_dir):
                    self.dir_list.append((target_dir, label_dir))
                else:
                    self.dir_list.append((process_target, process_dir_label))
                    break

    def match(self, targets, labels):
        # TODO: Extend for multi_modal
        matches = []
        for target in targets:
            target_head, target_tail = ntpath.split(target)
            for label in labels:
                label_head, label_tail = ntpath.split(label)
                if label_tail in target_tail or target_tail in label_tail:
                    matches.append((target, label))
                    break
        return matches

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        # Check if target dir exists else create
        mmcv.track_parallel_progress(self.convert_one,
                                     list(range(len(self.dir_list))),
                                     self.workers)
        print('\nFinished ...')

    def convert_one(self, directory_idx):
        """Convert action for files in directory.

        Args:
            directory List(String): Directory containing the files to be converted.
        """
        # Check if directory exists else create
        directory = self.dir_list[directory_idx]
        print("directory : ", directory)
        # TODO: Add Multi Modal creation, need to analyse calib file and get the matching files
        file_list2 = None
        file_list = glob(os.path.join(directory[0], '*'))
        if self.is_multi_modal:
            file_list2 = glob(os.path.join(directory[1], '*'))
            file_list = zip(file_list, file_list2)
        label_list = glob(os.path.join(directory[-1], '*'))
        for file_idx, (file, label) in enumerate(zip(file_list, label_list)):
            if self.is_multi_modal:
                pass
            else:
                if self.is_pcd:
                    self.save_lidar(directory_idx, file_idx, file)
            #    if self.is_img:
            #        self.save_image(directory_idx, file_idx, 1, file)
            if not self.test_mode:
                self.save_label(directory_idx, file_idx, label)

    def save_lidar(self, dir_idx, file_idx, file):
        """
        Convert file from .pcd to .bin and save
        Args:
            file: filename .pcd
        """

        point_cloud = pypcd.PointCloud.from_path(file)
        np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32) / 256
        point_cloud = None
        # concatenate x,y,z,intensity -> dim-4
        bin_format = np.column_stack((np_x, np_y, np_z, np_i))
        np_x = None
        np_y = None
        np_z = None
        np_i = None
        # bin_format.tofile(os.path.join(self.point_cloud_save_dir
        #                                f'{filename}.bin'))
        bin_format.tofile(os.path.join(self.point_cloud_save_dir,
                                       f'{str(dir_idx).zfill(3)}' +
                                       f'{str(file_idx).zfill(3)}.bin'))
        bin_format = None

    def save_image(self, directory_idx, file_idx, camera_idx, image):
        img_path = f'{self.image_save_dir}{str(camera_idx - 1)}/' + \
                   f'{str(file_idx).zfill(6)}' + \
                   f'{str(directory_idx).zfill(3)}.png'
        shutil.copyfile(image, img_path)

    def save_calib(self, directory_idx, file_idx, file):
        """

        """
        # TODO: Add when need multi modal data
        pass

    def save_label(self, directory_idx, file_idx, file):
        """
        Iterate through the labels for a given file and save it as kitti format
        Args:
            directory_idx:
            file_idx:
            file:
        """
        with open(file, 'r') as f:
            label_data = json.load(f)
        # fp_label_all = open(
        #     f'{self.label_all_save_dir}/{self.prefix}' +
        #     f'{str(file_idx).zfill(3)}{str(directory_idx).zfill(3)}.txt', 'w+')
        # TODO: Check json version
        label_version = 1
        for label in label_data['labels']:
            if 'box3d' in label:
                label_version = 1
                break
            else:
                label_version = 2
                break

        # Get bounding box
        lines = []
        id_to_box = dict()
        for label in label_data["labels"]:
            # pprint.pprint(label)
            if label_version == 1:
                x = label["box3d"]["location"]["x"]
                y = label["box3d"]["location"]["y"]
                z = label["box3d"]["location"]["z"]
                length = label["box3d"]["dimension"]["length"]
                width = label["box3d"]["dimension"]["width"]
                height = label["box3d"]["dimension"]["height"]
                heading = label["box3d"]["orientation"]["rotationYaw"]
                occluded = 0
            else:
                x = label["center"]["x"]
                y = label["center"]["y"]
                z = label["center"]["z"]
                height = label["dimensions"]["height"]
                length = label["dimensions"]["length"]
                width = label["dimensions"]["width"]
                heading = label["rotation"]["_z"]
                occluded = label["attributes"]["Occluded"]["value"]
            bounding_box = [
                x - length / 2,
                y - width / 2,
                x + length / 2,
                y + width / 2,
            ]
            id_to_box[label["id"]] = bounding_box

            # Not available
            truncated = 0
            alpha = -10
            occluded = 0
            # TODO: implement mapping for occluded
            z = z - height / 2
            line = f"{label['category']} {round(truncated, 2)} {occluded} {round(alpha, 2)} " + \
                   f"{round(bounding_box[0], 2)} {round(bounding_box[1], 2)} {round(bounding_box[2], 2)} " + \
                   f"{round(bounding_box[3], 2)} {round(height, 2)} {round(width, 2)} {round(length, 2)} " + \
                   f"{round(x, 2)} {round(y, 2)} {round(z, 2)} {round(heading, 2)}\n"

            # TODO: May consider tracking saving unique id
            # if self.save_track_id:
            #     line_all = line[:-1] + ' ' + file + ' ' + track_id + '\n'
            # else:
            #     line_all = line[:-1] + ' ' + name + '\n'
            lines.append(line)
            # TODO: Adapt to multi modal
        fp_label = open(
            f'{self.label_save_dir}{0}/' +
            f'{str(directory_idx).zfill(3)}{str(file_idx).zfill(3)}.txt', 'a')
        fp_label.writelines(lines)
        fp_label.close()

        # TODO: Adapt tracking later
        # fp_label_all.write(line_all)
        #
        # fp_label_all.close()

    def create_folder(self):
        """Create folder for data preprocessing."""
        # TODO: Make this version dependant 
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            # TODO: Make this dynamic based on cameras on intersection
            if self.is_multi_modal:
                for i in range(5):
                    mmcv.mkdir_or_exist(f'{d}{str(i)}')
            else:
                mmcv.mkdir_or_exist(f'{d}0')
