from typing import Any, Dict

import mmcv
import numpy as np
import torch
import torchvision
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
from numpy import random
from PIL import Image

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    box_np_ops,
)
from mmdet.datasets.builder import PIPELINES

from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines.data_augment_utils import noise_per_object_v3_

import os
# from mmdet3d.core.utils import visualize_lidar, visualize_lidar_two

from scipy.spatial.transform import Rotation


@PIPELINES.register_module()
class GlobalRotScaleTransCoop:

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict["infrastructure_points"].translate(trans_factor)
        input_dict["vehicle_points"].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        rot_mat_T = input_dict['vehicle_points'].rotate(noise_rotation)
        rot_mat_T_ = input_dict['infrastructure_points'].rotate(noise_rotation)
        np.testing.assert_array_almost_equal(rot_mat_T, rot_mat_T_)
        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = noise_rotation

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                input_dict[key].rotate(noise_rotation)

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']

        # vehicle_points
        points = input_dict['vehicle_points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['vehicle_points'] = points

        # infrastructure_points
        points = input_dict['infrastructure_points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['infrastructure_points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor



    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """

        # TODO : Check why gt_bboxes_3d are not augmented

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str

@PIPELINES.register_module()
class VehiclePointsToInfraCoords:
    def __call__(self, data):
        v2i = np.asarray(data["vehicle2infrastructure"])
        v2i_rot = v2i[:3, :3]
        v2i_trans = v2i[:3, 3]

        data["vehicle_points"].rotate(v2i_rot.T)
        data["vehicle_points"].translate(v2i_trans)    

        #if not self.is_train:
        # visualize_feature_lidar(data["vehicle_points"].tensor.numpy(), path="/home/cmt/viz_a9_featmap/features/lidar/vehicle/", file="pipelinev2i")
        # visualize_feature_lidar(data["infrastructure_points"].tensor.numpy(), path="/home/cmt/viz_a9_featmap/features/lidar/vehicle/", file="pipelineinfra")

        # vis_pts = [data["vehicle_points"].tensor.numpy(), data["infrastructure_points"].tensor.numpy()]
        # visualize_feature_lidar(vis_pts, path="/home/cmt/viz_a9_featmap/features/lidar/vehicle/", file="infra_vehicle")

        return data

@PIPELINES.register_module()
class TransformLidar2ImgToInfraCoords:

    def __call__(self, data):
        
        for i in range(len(data['vehicle_lidar2img'])):
            data['vehicle_lidar2img'][i] = data['vehicle_lidar2img'][i] @ np.linalg.inv(data['vehicle2infrastructure'])
            data['vehicle_lidar2cam'][i] = data['vehicle_lidar2cam'][i] @ np.linalg.inv(data['vehicle2infrastructure'])

        return data

    
@PIPELINES.register_module()
class PointsRangeFilterCoop(object):
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'vehicle_points' and 'infrastructure_points', \
                'pts_instance_mask' and 'pts_semantic_mask' keys are updated in the result dict.
        """

        if "vehicle_points" in data:
            points = data["vehicle_points"]
            points_mask = points.in_range_3d(self.pcd_range)
            clean_points = points[points_mask]
            data["vehicle_points"] = clean_points

        if "infrastructure_points" in data:
            points = data["infrastructure_points"]
            points_mask = points.in_range_3d(self.pcd_range)
            clean_points = points[points_mask]
            data["infrastructure_points"] = clean_points
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class ObjectSampleCoop(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
        use_ground_plane (bool): Whether to use gound plane to adjust the
            3D labels.
    """

    def __init__(self, db_sampler, sample_2d=False, use_ground_plane=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.use_ground_plane = use_ground_plane

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation,
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        if self.use_ground_plane and 'plane' in input_dict['ann_info']:
            ground_plane = input_dict['ann_info']['plane']
            input_dict['plane'] = ground_plane
        else:
            ground_plane = None

        # change to float for blending operation
        vehicle_points = input_dict['vehicle_points']
        infrastructure_points = input_dict['infrastructure_points']
        
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                img=None,
                ground_plane=ground_plane)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            infrastructure_points = self.remove_points_in_boxes(infrastructure_points, sampled_gt_bboxes_3d)
            # check the points dimension
            infrastructure_points = infrastructure_points.cat([sampled_points, infrastructure_points])
            
            vehicle_points = self.remove_points_in_boxes(vehicle_points, sampled_gt_bboxes_3d)
            # check the points dimension
            vehicle_points = vehicle_points.cat([sampled_points, vehicle_points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict["vehicle_points"] = vehicle_points
        input_dict["infrastructure_points"] = infrastructure_points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str

@PIPELINES.register_module()
class PointShuffleCoop:
    def __call__(self, data):
        data["vehicle_points"].shuffle()
        data["infrastructure_points"].shuffle()
        return data


@PIPELINES.register_module()
class PadMultiViewImageCoop(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""

        # img_keys = ['img']
        img_keys = ['vehicle_', 'infrastructure_']

        for k in img_keys:
            if self.size is not None:
                padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results[k + 'img']]
            elif self.size_divisor is not None:
                padded_img = [mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results[k + 'img']]
            results[k + 'img'] = padded_img

            results[k + 'img_shape'] = [img.shape for img in padded_img]
            results[k + 'pad_shape'] = [img.shape for img in padded_img]


        # TODO : Assume both infra and vehicular cam use same shape

        # results['img_shape'] = [img.shape for img in padded_img]
        # results['pad_shape'] = [img.shape for img in padded_img]

        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImageCoop(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # TODO : Seperate norm for vehicle and infra
        results['vehicle_img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['vehicle_img']]
        results['infrastructure_img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['infrastructure_img']]
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class UnifiedObjectSampleCoop(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, sample_method='depth', modify_points=False, mixup_rate=-1):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.sample_method = sample_method
        self.modify_points = modify_points
        self.mixup_rate = mixup_rate
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        # points = input_dict['points']
        vehicle_points = input_dict["vehicle_points"] # In infra coord
        infrastructure_points = input_dict["infrastructure_points"] # In infra coord

        if self.sample_2d:
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                with_img=True)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=False)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_points_idx = sampled_dict["points_idx"]
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            infrastructure_points = self.remove_points_in_boxes(infrastructure_points, sampled_gt_bboxes_3d)
            infrastructure_points_idx = -1 * np.ones(len(infrastructure_points), dtype=np.int)
            infrastructure_points = infrastructure_points.cat([infrastructure_points, sampled_points])
            infrastructure_points_idx = np.concatenate([infrastructure_points_idx, sampled_points_idx], axis=0)


            vehicle_points = self.remove_points_in_boxes(vehicle_points, sampled_gt_bboxes_3d)
            vehicle_points_idx = -1 * np.ones(len(vehicle_points), dtype=np.int)
            vehicle_points = vehicle_points.cat([vehicle_points, sampled_points])
            vehicle_points_idx = np.concatenate([vehicle_points_idx, sampled_points_idx], axis=0)

            # points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # points_idx = -1 * np.ones(len(points), dtype=np.int)
            # # check the points dimension
            # # points = points.cat([sampled_points, points])
            # points = points.cat([points, sampled_points])
            # points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)

            if self.sample_2d:

                sampled_img = sampled_dict['images']
                sampled_num = len(sampled_gt_bboxes_3d)

                infrastructure_imgs = input_dict['infrastructure_img']
                infrastructure_lidar2img = input_dict['infrastructure_lidar2img']
                infrastructure_imgs, infrastructure_points_keep = self.unified_sample(
                    infrastructure_imgs, infrastructure_lidar2img, 
                    infrastructure_points.tensor.numpy(), 
                    infrastructure_points_idx, 
                    gt_bboxes_3d.corners.numpy(), sampled_img, sampled_num)
                
                input_dict['infrastructure_img'] = infrastructure_imgs

                if self.modify_points:
                    infrastructure_points = infrastructure_points[infrastructure_points_keep]

                vehicle_imgs = input_dict['vehicle_img']
                vehicle_lidar2img = input_dict['vehicle_lidar2img']
                vehicle_imgs, vehicle_points_keep = self.unified_sample(
                    vehicle_imgs, vehicle_lidar2img, 
                    vehicle_points.tensor.numpy(), 
                    vehicle_points_idx, 
                    gt_bboxes_3d.corners.numpy(), sampled_img, sampled_num)
                
                input_dict['vehicle_img'] = vehicle_imgs

                if self.modify_points:
                    vehicle_points = vehicle_points[vehicle_points_keep]

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['infrastructure_points'] = infrastructure_points
        input_dict['vehicle_points'] = vehicle_points

        return input_dict

    def unified_sample(self, imgs, lidar2img, points, points_idx, bboxes_3d, sampled_img, sampled_num):
        # for boxes
        bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)
        is_raw = np.ones(len(bboxes_3d))
        is_raw[-sampled_num:] = 0
        is_raw = is_raw.astype(bool)
        raw_num = len(is_raw)-sampled_num
        # for point cloud
        points_3d = points[:,:4].copy()
        points_3d[:,-1] = 1
        points_keep = np.ones(len(points_3d)).astype(np.bool)
        new_imgs = imgs

        assert len(imgs)==len(lidar2img) and len(sampled_img)==sampled_num
        for _idx, (_img, _lidar2img) in enumerate(zip(imgs, lidar2img)):
            coord_img = bboxes_3d @ _lidar2img.T
            coord_img[...,:2] /= coord_img[...,2,None]
            depth = coord_img[...,2]
            img_mask = (depth > 0).all(axis=-1)
            img_count = img_mask.nonzero()[0]
            if img_mask.sum() == 0:
                continue
            depth = depth.mean(1)[img_mask]
            coord_img = coord_img[...,:2][img_mask]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=_img.shape[1]-1)
            bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=_img.shape[0]-1)
            img_mask = ((bbox[:,2:]-bbox[:,:2]) > 1).all(axis=-1)
            if img_mask.sum() == 0:
                continue
            depth = depth[img_mask]
            if 'depth' in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)
            img_count = img_count[img_mask][paste_order]
            bbox = bbox[img_mask][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int)
            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1]:_box[3],_box[0]:_box[2]])

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = raw_img.pop(0)
                    else:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = \
                            _img[_box[1]:_box[3],_box[0]:_box[2]] * (1 - self.mixup_rate) + raw_img.pop(0) * self.mixup_rate
                    fg_mask[_box[1]:_box[3],_box[0]:_box[2]] = 1
                else:
                    img_crop = sampled_img[_count-raw_num]
                    if len(img_crop)==0: continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2,3]]-_box[[0,1]]))
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = img_crop
                    else:
                        _img[_box[1]:_box[3],_box[0]:_box[2]] = \
                            _img[_box[1]:_box[3],_box[0]:_box[2]] * (1 - self.mixup_rate) + img_crop * self.mixup_rate

                paste_mask[_box[1]:_box[3],_box[0]:_box[2]] = _count
            
            new_imgs[_idx] = _img

            # calculate modify mask
            if self.modify_points:
                points_img = points_3d @ _lidar2img.T
                points_img[:,:2] /= points_img[:,2,None]
                depth = points_img[:,2]
                img_mask = depth > 0
                if img_mask.sum() == 0:
                    continue
                img_mask = (points_img[:,0] > 0) & (points_img[:,0] < _img.shape[1]) & \
                           (points_img[:,1] > 0) & (points_img[:,1] < _img.shape[0]) & img_mask
                points_img = points_img[img_mask].astype(int)
                new_mask = paste_mask[points_img[:,1], points_img[:,0]]==(points_idx[img_mask]+raw_num)
                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_num)
                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                raw_mask = raw_fg[points_img[:,1], points_img[:,0]] | raw_bg[points_img[:,1], points_img[:,0]]
                keep_mask = new_mask | raw_mask
                points_keep[img_mask] = points_keep[img_mask] & keep_mask

        return new_imgs, points_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipImageCoop(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True, pic_wise=False):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.pic_wise = pic_wise
        

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        # TODO : Repeat for vehicle

        infrastructure_imgs = results["infrastructure_img"]

        new_imgs = []
        new_depths = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i, img in enumerate(infrastructure_imgs):
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # augmentation (resize, crop, horizontal flip, rotate)
            if self.pic_wise:
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
            img, post_rot2, post_tran2 = self._img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            assert "depths" not in results.keys()
            if "depths" in results.keys():
                depth = results['depths'][i]
                depth = self._depth_transform(
                    depth,
                    resize=resize,
                    resize_dims=self.data_aug_conf["final_dim"],
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                new_depths.append(depth.astype(np.float32))

            new_imgs.append(img)
            results['infrastructure_cam_intrinsic'][i][:2, :3] = post_rot2 @ results['infrastructure_cam_intrinsic'][i][:2, :3]
            results['infrastructure_cam_intrinsic'][i][:2, 2] = post_tran2 + results['infrastructure_cam_intrinsic'][i][:2, 2]

        results["infrastructure_img"] = new_imgs
        # results["depths"] = new_depths
        results['infrastructure_lidar2img'] = [results['infrastructure_cam_intrinsic'][i] @ results['infrastructure_lidar2cam'][i] for i in range(len(results['infrastructure_lidar2cam']))]


        vehicle_imgs = results["vehicle_img"]

        new_imgs = []
        new_depths = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i, img in enumerate(vehicle_imgs):
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # augmentation (resize, crop, horizontal flip, rotate)
            if self.pic_wise:
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
            img, post_rot2, post_tran2 = self._img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            assert "depths" not in results.keys()
            if "depths" in results.keys():
                depth = results['depths'][i]
                depth = self._depth_transform(
                    depth,
                    resize=resize,
                    resize_dims=self.data_aug_conf["final_dim"],
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                new_depths.append(depth.astype(np.float32))

            new_imgs.append(img)
            results['vehicle_cam_intrinsic'][i][:2, :3] = post_rot2 @ results['vehicle_cam_intrinsic'][i][:2, :3]
            results['vehicle_cam_intrinsic'][i][:2, 2] = post_tran2 + results['vehicle_cam_intrinsic'][i][:2, 2]

        results["vehicle_img"] = new_imgs
        # results["depths"] = new_depths
        results['vehicle_lidar2img'] = [results['vehicle_cam_intrinsic'][i] @ results['vehicle_lidar2cam'][i] for i in range(len(results['vehicle_lidar2cam']))]

        return results

    def _get_rot(self, h):

        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
        # adjust image
        resized_img = cv2.resize(img, resize_dims)
        img = np.zeros((crop[3] - crop[1], crop[2] - crop[0], 3))
        
        hsize, wsize = crop[3] - crop[1], crop[2] - crop[0]
        dh, dw, sh, sw = crop[1], crop[0], 0, 0
        
        if dh < 0:
            sh = -dh
            hsize += dh
            dh = 0
        if dh + hsize > resized_img.shape[0]:
            hsize = resized_img.shape[0] - dh
        if dw < 0:
            sw = -dw
            wsize += dw
            dw = 0
        if dw + wsize > resized_img.shape[1]:
            wsize = resized_img.shape[1] - dw
        img[sh : sh + hsize, sw : sw + wsize] = resized_img[dh: dh + hsize, dw: dw + wsize]
        
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        if flip:
            img = cv2.flip(img, 1)
        M = cv2.getRotationMatrix2D(center, rotate, scale=1.0)
        img = cv2.warpAffine(img, M, (w, h))
        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _depth_transform(self, cam_depth, resize, resize_dims, crop, flip, rotate):
        """
        Input:
            cam_depth: Nx3, 3: x,y,d
            resize: a float value
            resize_dims: self.ida_aug_conf["final_dim"] -> [H, W]
            crop: x1, y1, x2, y2
            flip: bool value
            rotate: an angle
        Output:
            cam_depth: [h/down_ratio, w/down_ratio, d]
        """

        H, W = resize_dims
        cam_depth[:, :2] = cam_depth[:, :2] * resize
        cam_depth[:, 0] -= crop[0]
        cam_depth[:, 1] -= crop[1]
        if flip:
            cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

        cam_depth[:, 0] -= W / 2.0
        cam_depth[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

        cam_depth[:, 0] += W / 2.0
        cam_depth[:, 1] += H / 2.0

        depth_coords = cam_depth[:, :2].astype(np.int16)

        depth_map = np.zeros((H, W, 3))
        valid_mask = (
            (depth_coords[:, 1] < resize_dims[0])
            & (depth_coords[:, 0] < resize_dims[1])
            & (depth_coords[:, 1] >= 0)
            & (depth_coords[:, 0] >= 0)
        )
        depth_map[depth_coords[valid_mask, 1], depth_coords[valid_mask, 0], :] = cam_depth[valid_mask, :]

        return depth_map


@PIPELINES.register_module()
class GlobalRotScaleTransAllCoop(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        
        # input_dict['points'].translate(trans_factor)
        # if 'radar' in input_dict:
        #     input_dict['radar'].translate(trans_factor)
        input_dict["infrastructure_points"].translate(trans_factor)
        input_dict["vehicle_points"].translate(trans_factor)

        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

        trans_mat = np.eye(4)
        trans_mat[:3, -1] = trans_factor
        trans_mat_inv = np.linalg.inv(trans_mat)

        for view in range(len(input_dict["infrastructure_lidar2img"])):
            input_dict["infrastructure_lidar2img"][view] = input_dict["infrastructure_lidar2img"][view] @ trans_mat_inv
            input_dict["infrastructure_lidar2cam"][view] = input_dict["infrastructure_lidar2cam"][view] @ trans_mat_inv

        for view in range(len(input_dict["vehicle_lidar2img"])):
            input_dict["vehicle_lidar2img"][view] = input_dict["vehicle_lidar2img"][view] @ trans_mat_inv
            input_dict["vehicle_lidar2cam"][view] = input_dict["vehicle_lidar2cam"][view] @ trans_mat_inv

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if 'rot_degree' not in input_dict:
            rotation = self.rot_range
            noise_rotation = np.random.uniform(rotation[0], rotation[1])
        else:
            noise_rotation = input_dict['rot_degree']

        # if no bbox in input_dict, only rotate points
        if len(input_dict['bbox3d_fields']) == 0:
            if 'rot_degree' not in input_dict:
                rot_mat_T = input_dict['vehicle_points'].rotate(noise_rotation)
                rot_mat_T = input_dict['infrastructure_points'].rotate(noise_rotation)
                # if 'radar' in input_dict:
                #     input_dict['radar'].rotate(noise_rotation)
            else:
                rot_mat_T = input_dict['vehicle_points'].rotate(-noise_rotation)
                rot_mat_T = input_dict['infrastructure_points'].rotate(-noise_rotation)
                # if 'radar' in input_dict:
                #     input_dict['radar'].rotate(-noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T

            rot_mat = torch.eye(4)
            rot_mat[:3, :3].copy_(rot_mat_T)
            rot_mat[0, 1], rot_mat[1, 0] = -rot_mat[0, 1], -rot_mat[1, 0]
            rot_mat_inv = torch.inverse(rot_mat)
            for view in range(len(input_dict["vehicle_lidar2img"])):
                input_dict["vehicle_lidar2img"][view] = (torch.tensor(input_dict["vehicle_lidar2img"][view]).float() @ rot_mat_inv).numpy()
                input_dict["vehicle_lidar2cam"][view] = (torch.tensor(input_dict["vehicle_lidar2cam"][view]).float() @ rot_mat_inv).numpy()

            for view in range(len(input_dict["infrastructure_lidar2img"])):
                input_dict["infrastructure_lidar2img"][view] = (torch.tensor(input_dict["infrastructure_lidar2img"][view]).float() @ rot_mat_inv).numpy()
                input_dict["infrastructure_lidar2cam"][view] = (torch.tensor(input_dict["infrastructure_lidar2cam"][view]).float() @ rot_mat_inv).numpy()
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:

                # rotate gt_bboxes_3d , and infrastructure_points
                infrastructure_points, rot_mat_T = input_dict[key].rotate(noise_rotation, input_dict['infrastructure_points'])
                input_dict['infrastructure_points'] = infrastructure_points
                input_dict['pcd_rotation'] = rot_mat_T
                input_dict['pcd_rotation_angle'] = noise_rotation

                # rotate vehicle_points , and infrastructure_points
                rot_mat_T_ = input_dict['vehicle_points'].rotate(noise_rotation)

                # vehicle_points, rot_mat_T = input_dict[key].rotate(noise_rotation, input_dict['vehicle_points'])
                # input_dict['vehicle_points'] = vehicle_points

                np.testing.assert_array_almost_equal(rot_mat_T, rot_mat_T_)



                # if 'radar' in input_dict:
                #     input_dict['radar'].rotate(-noise_rotation)

                rot_mat = torch.eye(4)
                rot_mat[:3, :3].copy_(rot_mat_T)
                rot_mat[0, 1], rot_mat[1, 0] = -rot_mat[0, 1], -rot_mat[1, 0]
                rot_mat_inv = torch.inverse(rot_mat)

                for view in range(len(input_dict["vehicle_lidar2img"])):
                    input_dict["vehicle_lidar2img"][view] = (torch.tensor(input_dict["vehicle_lidar2img"][view]).float() @ rot_mat_inv).numpy()
                    input_dict["vehicle_lidar2cam"][view] = (torch.tensor(input_dict["vehicle_lidar2cam"][view]).float() @ rot_mat_inv).numpy()

                for view in range(len(input_dict["infrastructure_lidar2img"])):
                    input_dict["infrastructure_lidar2img"][view] = (torch.tensor(input_dict["infrastructure_lidar2img"][view]).float() @ rot_mat_inv).numpy()
                    input_dict["infrastructure_lidar2cam"][view] = (torch.tensor(input_dict["infrastructure_lidar2cam"][view]).float() @ rot_mat_inv).numpy()


    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']

        # vehicle_points
        points = input_dict['vehicle_points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['vehicle_points'] = points

        # infrastructure_points
        points = input_dict['infrastructure_points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['infrastructure_points'] = points
        
        # if 'radar' in input_dict:
        #     input_dict['radar'].scale(scale)
            
        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

        scale_mat = torch.tensor(
            [
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, scale, 0],
                [0, 0, 0, 1],
            ]
        )
        scale_mat_inv = torch.inverse(scale_mat)
        for view in range(len(input_dict["vehicle_lidar2img"])):
            input_dict["vehicle_lidar2img"][view] = (torch.tensor(input_dict["vehicle_lidar2img"][view]).float() @ scale_mat_inv).numpy()
            input_dict["vehicle_lidar2cam"][view] = (torch.tensor(input_dict["vehicle_lidar2cam"][view]).float() @ scale_mat_inv).numpy()
        for view in range(len(input_dict["infrastructure_lidar2img"])):
            input_dict["infrastructure_lidar2img"][view] = (torch.tensor(input_dict["infrastructure_lidar2img"][view]).float() @ scale_mat_inv).numpy()
            input_dict["infrastructure_lidar2cam"][view] = (torch.tensor(input_dict["infrastructure_lidar2cam"][view]).float() @ scale_mat_inv).numpy()

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTransImageCoop(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        shift_height=False,
        reverse_angle=False,
        # training=True,
        # flip_dx_ratio=0.0,
        # flip_dy_ratio=0.0
    ):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

        self.reverse_angle = reverse_angle

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        # random rotate
        self.rotate_bev_along_z(results)

        # if self.reverse_angle:
        #     rot_angle *= -1
        # results["gt_bboxes_3d"].rotate(
        #     np.array(rot_angle)
        # )  # mmdet LiDARInstance3DBoxes存的角度方向是反的(rotate函数实现的是绕着z轴由y向x转)

        # random scale
        # scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results)
        # results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        # self.flip_xy(results)

        return results

    def rotate_bev_along_z(self, input_dict):

        assert 'rot_degree' not in input_dict

        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # rotate gt_bboxes_3d
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:

                input_dict[key].rotate(noise_rotation)
                # input_dict['pcd_rotation'] = rot_mat_T
                # input_dict['pcd_rotation_angle'] = noise_rotation

        rot_cos = torch.cos(torch.tensor(noise_rotation))
        rot_sin = torch.sin(torch.tensor(noise_rotation))

        rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)
        for view in range(len(input_dict["vehicle_lidar2img"])):
            input_dict["vehicle_lidar2img"][view] = (torch.tensor(input_dict["vehicle_lidar2img"][view]).float() @ rot_mat_inv).numpy()
            input_dict["vehicle_lidar2cam"][view] = (torch.tensor(input_dict["vehicle_lidar2cam"][view]).float() @ rot_mat_inv).numpy()

        for view in range(len(input_dict["infrastructure_lidar2img"])):
            input_dict["infrastructure_lidar2img"][view] = (torch.tensor(input_dict["infrastructure_lidar2img"][view]).float() @ rot_mat_inv).numpy()
            input_dict["infrastructure_lidar2cam"][view] = (torch.tensor(input_dict["infrastructure_lidar2cam"][view]).float() @ rot_mat_inv).numpy()
        return

    def scale_xyz(self, input_dict):

        assert not self.shift_height

        scale = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
            
        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

        scale_mat = torch.tensor(
            [
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, scale, 0],
                [0, 0, 0, 1],
            ]
        )
        scale_mat_inv = torch.inverse(scale_mat)
        for view in range(len(input_dict["vehicle_lidar2img"])):
            input_dict["vehicle_lidar2img"][view] = (torch.tensor(input_dict["vehicle_lidar2img"][view]).float() @ scale_mat_inv).numpy()
            input_dict["vehicle_lidar2cam"][view] = (torch.tensor(input_dict["vehicle_lidar2cam"][view]).float() @ scale_mat_inv).numpy()
        for view in range(len(input_dict["infrastructure_lidar2img"])):
            input_dict["infrastructure_lidar2img"][view] = (torch.tensor(input_dict["infrastructure_lidar2img"][view]).float() @ scale_mat_inv).numpy()
            input_dict["infrastructure_lidar2cam"][view] = (torch.tensor(input_dict["infrastructure_lidar2cam"][view]).float() @ scale_mat_inv).numpy()

        return

    def flip_xy(self, results):
        mat = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
        if np.random.rand() < self.flip_dx_ratio:
            mat[0][0] = -1
            results["gt_bboxes_3d"].flip(bev_direction='vertical')
        if np.random.rand() < self.flip_dy_ratio:
            mat[1][1] = -1
            results["gt_bboxes_3d"].flip(bev_direction='horizontal')
            
        num_view = len(results['lidar2img'])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ mat.float()).numpy()
            results["lidar2cam"][view] = (torch.tensor(results["lidar2cam"][view]).float() @ mat.float()).numpy()
        return
