import os
from typing import Any, Dict, Tuple

import mmcv
import numpy as np
from PIL import Image


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams


@PIPELINES.register_module()
class LoadMultiViewImageFromFilesCoop:
    """Load multi channel images from a list of separate channel files.

    Expects results['vehicle_img_filename'] and results['infrastructure_img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        vehicle_filename = results['vehicle_img_filename']
        # img is of shape (h, w, c, num_views) = (1200, 1920, 3, 1)
        vehicle_img = np.stack(
            [mmcv.imread(name, self.color_type) for name in vehicle_filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['vehicle_filename'] = vehicle_filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['vehicle_img'] = [vehicle_img[..., i] for i in range(vehicle_img.shape[-1])]
        results['vehicle_img_shape'] = vehicle_img.shape
        results['vehicle_ori_shape'] = vehicle_img.shape

        infrastructure_filename = results['infrastructure_img_filename']
        # img is of shape (h, w, c, num_views) = (1200, 1920, 3, 3)
        infrastructure_img = np.stack(
            [mmcv.imread(name, self.color_type) for name in infrastructure_filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['infrastructure_filename'] = infrastructure_filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['infrastructure_img'] = [infrastructure_img[..., i] for i in range(infrastructure_img.shape[-1])]
        results['infrastructure_img_shape'] = infrastructure_img.shape
        results['infrastructure_ori_shape'] = infrastructure_img.shape


        # Assume image shapes are same for both infra and vehicle
        assert results['vehicle_img_shape'][:-1] == results['infrastructure_img_shape'][:-1]

        # Set initial values for default meta_keys
        results['vehicle_pad_shape'] = vehicle_img.shape
        results['infrastructure_pad_shape'] = infrastructure_img.shape

        # TODO : Seperate for vehicle and infrastructure
        results['scale_factor'] = 1.0
        num_channels = 1 if len(infrastructure_img.shape) < 3 else infrastructure_img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        


        return results


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromMultiSweepsCoop(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        time_dim=4,
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = False
        self.reduce_beams = False
        self.training = False

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        vehicle_points = results["vehicle_points"]
        vehicle_points.tensor[:, self.time_dim] = 0
        vehicle_sweep_points_list = [vehicle_points]
        vehicle_ts = results["timestamp"] / 1e6

        infrastructure_points = results["infrastructure_points"]
        infrastructure_points.tensor[:, self.time_dim] = 0
        infrastructure_sweep_points_list = [infrastructure_points]
        infrastructure_ts = results["timestamp"] / 1e6
        
        if self.pad_empty_sweeps and len(results["vehicle_sweeps"]) == 0 and len(results["infrastructure_sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    vehicle_sweep_points_list.append(self._remove_close(vehicle_points))
                    infrastructure_sweep_points_list.append(self._remove_close(infrastructure_points))
                else:
                    vehicle_sweep_points_list.append(vehicle_points)
                    infrastructure_sweep_points_list.append(infrastructure_points)
        else:
            if len(results["vehicle_sweeps"]) <= self.sweeps_num and len(results["infrastructure_sweeps"]) <= self.sweeps_num:
                vehicle_choices = np.arange(len(results["vehicle_sweeps"]))
                infrastructure_choices = np.arange(len(results["infrastructure_sweeps"]))
            elif self.test_mode:
                vehicle_choices = np.arange(self.sweeps_num)
                infrastructure_choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    vehicle_choices = np.random.choice(
                        len(results["vehicle_sweeps"]), self.sweeps_num, replace=False
                    )
                    infrastructure_choices = np.random.choice(
                        len(results["infrastructure_sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    vehicle_choices = np.random.choice(
                        len(results["vehicle_sweeps"]) - 1, self.sweeps_num, replace=False
                    )
                    infrastructure_choices = np.random.choice(
                        len(results["infrastructure_sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in vehicle_choices:
                vehicle_sweep = results["vehicle_sweeps"][idx]
                vehicle_points_sweep = self._load_points(vehicle_sweep["data_path"])
                vehicle_points_sweep = np.copy(vehicle_points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    vehicle_points_sweep = reduce_LiDAR_beams(vehicle_points_sweep, self.reduce_beams)

                if self.remove_close:
                    vehicle_points_sweep = self._remove_close(vehicle_points_sweep)
                vehicle_sweep_ts = vehicle_sweep["timestamp"] / 1e6
                vehicle_points_sweep[:, :3] = (
                    vehicle_points_sweep[:, :3] @ vehicle_sweep["sensor2lidar_rotation"].T
                )
                vehicle_points_sweep[:, :3] += vehicle_sweep["sensor2lidar_translation"]
                vehicle_points_sweep[:, 4] = vehicle_ts - vehicle_sweep_ts
                vehicle_points_sweep = vehicle_points.new_point(vehicle_points_sweep)
                vehicle_sweep_points_list.append(vehicle_points_sweep)
            
            for idy in infrastructure_choices:
                infrastructure_sweep = results["infrastructure_sweeps"][idy]
                infrastructure_points_sweep = self._load_points(infrastructure_sweep["data_path"])
                infrastructure_points_sweep = np.copy(infrastructure_points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    infrastructure_points_sweep = reduce_LiDAR_beams(infrastructure_points_sweep, self.reduce_beams)

                if self.remove_close:
                    infrastructure_points_sweep = self._remove_close(infrastructure_points_sweep)
                infrastructure_sweep_ts = infrastructure_sweep["timestamp"] / 1e6
                infrastructure_points_sweep[:, :3] = (
                    infrastructure_points_sweep[:, :3] @ infrastructure_sweep["sensor2lidar_rotation"].T
                )
                infrastructure_points_sweep[:, :3] += infrastructure_sweep["sensor2lidar_translation"]
                infrastructure_points_sweep[:, 4] = infrastructure_ts - infrastructure_sweep_ts
                infrastructure_points_sweep = infrastructure_points.new_point(infrastructure_points_sweep)
                infrastructure_sweep_points_list.append(infrastructure_points_sweep)

        vehicle_points = vehicle_points.cat(vehicle_sweep_points_list)
        vehicle_points = vehicle_points[:, self.use_dim]
        results["vehicle_points"] = vehicle_points
        infrastructure_points = infrastructure_points.cat(infrastructure_sweep_points_list)
        infrastructure_points = infrastructure_points[:, self.use_dim]
        results["infrastructure_points"] = infrastructure_points

        #if not self.training:
        #    visualize_feature_lidar(vehicle_points, "/home/bevfusion/viz_a9_featmap/features/lidar/vehicle/val_multisweep")
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@PIPELINES.register_module()
class LoadPointsFromFileCoop:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=False,
        reduce_beams=False,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        vehicle_lidar_path = results["vehicle_lidar_path"]
        infrastructure_lidar_path = results["infrastructure_lidar_path"]
        vehicle_points = self._load_points(vehicle_lidar_path)
        infrastructure_points = self._load_points(infrastructure_lidar_path)
        vehicle_points = vehicle_points.reshape(-1, self.load_dim)
        infrastructure_points = infrastructure_points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            vehicle_points = reduce_LiDAR_beams(vehicle_points, self.reduce_beams)
            infrastructure_points = reduce_LiDAR_beams(infrastructure_points, self.reduce_beams)
        vehicle_points = vehicle_points[:, self.use_dim]
        infrastructure_points = infrastructure_points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            vehicle_floor_height = np.percentile(vehicle_points[:, 2], 0.99)
            infrastructure_floor_height = np.percentile(infrastructure_points[:, 2], 0.99)
            vehicle_height = vehicle_points[:, 2] - vehicle_floor_height
            infrastructure_height = infrastructure_points[:, 2] - infrastructure_floor_height
            vehicle_points = np.concatenate(
                [vehicle_points[:, :3], np.expand_dims(vehicle_height, 1), vehicle_points[:, 3:]], 1
            )
            infrastructure_points = np.concatenate(
                [infrastructure_points[:, :3], np.expand_dims(infrastructure_height, 1), infrastructure_points[:, 3:]], 1
            )
            attribute_dims = dict(vehicle_height=3, infrastructure_height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    vehicle_color=[
                        vehicle_points.shape[1] - 3,
                        vehicle_points.shape[1] - 2,
                        vehicle_points.shape[1] - 1,
                    ],
                    infrastructure_color=[
                        infrastructure_points.shape[1] - 3,
                        infrastructure_points.shape[1] - 2,
                        infrastructure_points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        vehicle_points = points_class(
            vehicle_points, points_dim=vehicle_points.shape[-1], attribute_dims=attribute_dims
        )
        results["vehicle_points"] = vehicle_points
        infrastructure_points = points_class(
            infrastructure_points, points_dim=infrastructure_points.shape[-1], attribute_dims=attribute_dims
        )
        results["infrastructure_points"] = infrastructure_points

        #if not self.training:
        #    visualize_feature_lidar(vehicle_points, "/home/bevfusion/viz_a9_featmap/features/lidar/vehicle/val_loadfromfile")

        return results

