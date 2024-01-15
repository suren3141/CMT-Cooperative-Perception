import warnings
from os import path as osp

import mmcv
import torch
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet.core import multi_apply
# from .. import builder

from mmdet3d.models import builder
from mmdet.models import DETECTORS
from .coop_base import BaseCoop3DDetector

from projects.mmdet3d_plugin.models.detectors import CmtDetector

@DETECTORS.register_module()
class CmtCoopDetector(BaseCoop3DDetector):

    def __init__(self,
                 vehicle_model = None,
                 infrastructure_model = None,
                 coop_fusion_neck = None,
                #  pts_voxel_layer=None,
                #  pts_voxel_encoder=None,
                #  pts_middle_encoder=None,
                #  pts_fusion_layer=None,
                #  img_backbone=None,
                #  pts_backbone=None,
                #  img_neck=None,
                #  pts_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(BaseCoop3DDetector, self).__init__(init_cfg=init_cfg)

        if vehicle_model:
            self.vehicle_model = builder.build_model(vehicle_model)

        if infrastructure_model:
            self.infrastructure_model = builder.build_model(infrastructure_model)

        self.coop_fusion_neck = None
        if coop_fusion_neck:
            # TODO
            # self.coop_fusion_neck = CmtDetector(infrastructure_model)
            raise NotImplementedError


        '''
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(
                pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        '''

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg.')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)
        if self.with_img_roi_head:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg.')
                self.img_roi_head.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)
        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg')
                self.pts_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=pts_pretrained)
                
    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None


    @property
    def with_vehicle_model(self):
        """bool: Whether the detector has a vehicle_model."""
        return hasattr(self,
                       'vehicle_model') and self.vehicle_model is not None

    @property
    def with_infrastructure_model(self):
        """bool: Whether the detector has a infrastructure_model."""
        return hasattr(self,
                       'infrastructure_model') and self.infrastructure_model is not None
    

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
        
    #     def load(module, prefix=''):
    #         # recursively check parallel module in case that the model has a
    #         # complicated structure, e.g., nn.Module(nn.Module(DDP))
    #         module._load_from_state_dict(state_dict, prefix, local_metadata, strict,
    #                                     missing_keys, unexpected_keys,
    #                                     error_msgs)
    #         for name, child in module._modules.items():
    #             if child is not None:
    #                 load(child, prefix + name + '.')

    #     """Overload in order to load img network ckpts into img branch."""
    #     for name, child in self._modules.items():
    #         if name in ['vehicle_model', 'infrastructure_model'] and child is not None:
    #             load(child, prefix)

        # module_names = ['backbone', 'neck', 'roi_head', 'rpn_head']
        # for key in list(state_dict):
        #     for module_name in module_names:
        #         if key.startswith(module_name) and ('img_' +
        #                                             key) not in state_dict:
        #             state_dict['img_' + key] = state_dict.pop(key)

        # super()._load_from_state_dict(state_dict, prefix, local_metadata,
        #                               strict, missing_keys, unexpected_keys,
        #                               error_msgs)

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        raise NotImplementedError
        # if self.with_img_backbone and img is not None:
        #     input_shape = img.shape[-2:]
        #     # update real input shape of each single img
        #     for img_meta in img_metas:
        #         img_meta.update(input_shape=input_shape)

        #     if img.dim() == 5 and img.size(0) == 1:
        #         img.squeeze_()
        #     elif img.dim() == 5 and img.size(0) > 1:
        #         B, N, C, H, W = img.size()
        #         img = img.view(B * N, C, H, W)
        #     img_feats = self.img_backbone(img)
        # else:
        #     return None
        # if self.with_img_neck:
        #     img_feats = self.img_neck(img_feats)
        # return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        raise NotImplementedError
        # if not self.with_pts_bbox:
        #     return None
        # voxels, num_points, coors = self.voxelize(pts)
        # voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
        #                                         img_feats, img_metas)
        # batch_size = coors[-1, 0] + 1
        # x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # x = self.pts_backbone(x)
        # if self.with_pts_neck:
        #     x = self.pts_neck(x)
        # return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        raise NotImplementedError
        # img_feats = self.extract_img_feat(img, img_metas)
        # pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        # return (img_feats, pts_feats)

    
    def extract_vehicular_feat(self, vehicle_points, vehicle_img, vehicle_img_metas):
        if self.with_vehicle_model:
            vehicle_img_feats, vehicle_pts_feats = self.vehicle_model.extract_feat(points=vehicle_points, img=vehicle_img, img_metas=vehicle_img_metas)
        else:
            vehicle_img_feats, vehicle_pts_feats = None, None
        return vehicle_img_feats, vehicle_pts_feats

    def extract_infrastructure_feat(self, infrastructure_points, infrastructure_img, infrastructure_img_metas):
        if self.with_infrastructure_model:
            infrastructure_img_feats, infrastructure_pts_feats = self.infrastructure_model.extract_feat(points=infrastructure_points, img=infrastructure_img, img_metas=infrastructure_img_metas)
        else:
            infrastructure_img_feats, infrastructure_pts_feats = None, None
        return infrastructure_img_feats, infrastructure_pts_feats

    def coop_fuse_pts_features(self, vehicle_pts_feats, infrastructure_pts_feats):
        """stack features along new axis and get maximum"""
        if vehicle_pts_feats is None and infrastructure_pts_feats is None:
            return None
        if vehicle_pts_feats is None:
            return infrastructure_pts_feats
        if infrastructure_pts_feats is None:
            return vehicle_pts_feats

        assert len(vehicle_pts_feats) == len(infrastructure_pts_feats) == 1
        pts_feats = torch.stack([vehicle_pts_feats[0], infrastructure_pts_feats[0]])
        return [torch.max(pts_feats, 0).values]
    
    def coop_fuse_img_features(self, vehicle_img_feats, infrastructure_img_feats):

        if infrastructure_img_feats is None:
            return vehicle_img_feats
        
        return infrastructure_img_feats

        assert len(vehicle_img_feats) == len(infrastructure_img_feats)

        img_feats = [torch.stack([v_feat, i_feat]) for (v_feat, i_feat) in zip(vehicle_img_feats, infrastructure_img_feats)]
        [torch.max(img_feats, 0).values]
        return None


    def forward_train(self,
                      vehicle_points=None,
                      vehicle_img=None,
                      infrastructure_points=None,
                      infrastructure_img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        vehicle_img_feats, vehicle_pts_feats = self.extract_vehicular_feat(vehicle_points, vehicle_img, img_metas)
        infrastructure_img_feats, infrastructure_pts_feats = self.extract_infrastructure_feat(infrastructure_points, infrastructure_img, img_metas)

        losses = dict()
        if vehicle_img_feats or vehicle_pts_feats or infrastructure_pts_feats or infrastructure_img_feats:
            losses_pts = self.coop_forward_pts_train(vehicle_pts_feats, vehicle_img_feats,
                                                    infrastructure_pts_feats, infrastructure_img_feats,
                                                    gt_bboxes_3d, gt_labels_3d, img_metas,
                                                    gt_bboxes_ignore)
        
            losses.update(losses_pts)

        # if infrastructure_pts_feats or infrastructure_img_feats:
        #     losses_pts = self.infrastructure_forward_pts_train(infrastructure_pts_feats, infrastructure_img_feats,
        #                                                        gt_bboxes_3d, gt_labels_3d, img_metas,
        #                                                        gt_bboxes_ignore)
        #     losses.update(losses_pts)

        # if vehicle_pts_feats or vehicle_img_feats:
        #     losses_pts = self.vehicle_forward_pts_train(vehicle_pts_feats, vehicle_img_feats,
        #                                                 gt_bboxes_3d, gt_labels_3d, img_metas,
        #                                                 gt_bboxes_ignore)
        #     losses.update(losses_pts)

        # pts_feats = self.coop_fuse_pts_features(vehicle_pts_feats, infrastructure_pts_feats)
        # img_feats = self.coop_fuse_img_features(vehicle_img_feats, infrastructure_img_feats)

        # losses = dict()
        # if pts_feats or img_feats:
        #     losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
        #                                         gt_labels_3d, img_metas,
        #                                         gt_bboxes_ignore)
        #     losses.update(losses_pts)
        return losses
    
    def filter_img_metas(self, img_meta, prefix='', ignore=''):

        filt_img_meta = dict()

        for k, v in img_meta.items():
            if k.startswith(prefix):
                k_new = k[len(prefix):]
                filt_img_meta[k_new] = v
            elif not k.startswith(ignore):
                filt_img_meta[k] = v

        filt_img_meta['node'] = prefix

        return filt_img_meta



    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def infrastructure_forward_pts_train(self,
                                        pts_feats,
                                        img_feats,
                                        gt_bboxes_3d,
                                        gt_labels_3d,
                                        img_metas,
                                        gt_bboxes_ignore=None):
        """Forward function for infrastructure branch."""

        infrastructure_img_metas = [self.filter_img_metas(img_meta, prefix='infrastructure_', ignore='vehicle_') for img_meta in img_metas]

        return self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d, gt_labels_3d, infrastructure_img_metas, gt_bboxes_ignore=gt_bboxes_ignore)        

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def vehicle_forward_pts_train(self,
                                pts_feats,
                                img_feats,
                                gt_bboxes_3d,
                                gt_labels_3d,
                                img_metas,
                                gt_bboxes_ignore=None):
        """Forward function for infrastructure branch."""

        vehicle_img_metas = [self.filter_img_metas(img_meta, prefix='vehicle_', ignore='infrastructure_') for img_meta in img_metas]

        return self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d, gt_labels_3d, vehicle_img_metas, gt_bboxes_ignore=gt_bboxes_ignore)        


    @force_fp32(apply_to=('vehicle_pts_feats', 'vehicle_img_feats', 'infrastructure_pts_feats', 'infrastructure_img_feats'))
    def coop_forward_pts_train(self,
                        vehicle_pts_feats, vehicle_img_feats,
                        infrastructure_pts_feats, infrastructure_img_feats,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        img_metas,
                        gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if vehicle_pts_feats is None:
            vehicle_pts_feats = [None]
        if infrastructure_pts_feats is None:
            infrastructure_pts_feats = [None]
        if vehicle_img_feats is None:
            vehicle_img_feats = [None]
        if infrastructure_img_feats is None:
            infrastructure_img_feats = [None]

        outs = self.pts_bbox_head(vehicle_pts_feats, infrastructure_pts_feats, vehicle_img_feats, infrastructure_img_feats, img_metas)
        # loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        # losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses


    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        # loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        # losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self,
                     vehicle_points=None,
                     vehicle_img=None,
                     infrastructure_points=None,
                     infrastructure_img=None,
                     img_metas=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if vehicle_points is None: vehicle_points = [None]
        if infrastructure_points is None: infrastructure_points = [None]
        if vehicle_img is None: vehicle_img = [None]
        if infrastructure_img is None: infrastructure_img = [None]

        for var, name in [(vehicle_points, 'vehicle_points'), (vehicle_img, 'vehicle_img'), (infrastructure_points, 'infrastructure_points'), (infrastructure_img, 'infrastructure_img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))

        return self.simple_test(vehicle_points[0], infrastructure_points[0], img_metas[0], vehicle_img=vehicle_img[0], infrastructure_img=infrastructure_img[0], **kwargs)
    
    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results

    @force_fp32(apply_to=('infrastructure_pts_feats', 'infrastructure_pts_feats', 'vehicle_img_feats', 'infrastructure_img_feats'))
    def coop_simple_test_pts(self, vehicle_pts_feats, infrastructure_pts_feats, vehicle_img_feats, infrastructure_img_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        if vehicle_pts_feats is None:
            vehicle_pts_feats = [None]
        if infrastructure_pts_feats is None:
            infrastructure_pts_feats = [None]
        if vehicle_img_feats is None:
            vehicle_img_feats = [None]
        if infrastructure_img_feats is None:
            infrastructure_img_feats = [None]

        outs = self.pts_bbox_head(vehicle_pts_feats, infrastructure_pts_feats, vehicle_img_feats, infrastructure_img_feats, img_metas)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results

    def simple_test(self, 
                    vehicle_points,
                    infrastructure_points,
                    img_metas,
                    vehicle_img=None,
                    infrastructure_img=None,
                    rescale=False):
        
        vehicle_img_feats, vehicle_pts_feats = self.extract_vehicular_feat(vehicle_points, vehicle_img, img_metas)
        infrastructure_img_feats, infrastructure_pts_feats = self.extract_infrastructure_feat(infrastructure_points, infrastructure_img, img_metas)
        
        bbox_list = [dict() for i in range(len(img_metas))]
        if self.with_pts_bbox:
            bbox_pts = self.coop_simple_test_pts(vehicle_pts_feats, infrastructure_pts_feats, vehicle_img_feats, infrastructure_img_feats, img_metas, rescale=rescale)
            
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox

        # if img_feats and self.with_img_bbox:
        #     bbox_img = self.simple_test_img(
        #         img_feats, img_metas, rescale=rescale)
        #     for result_dict, img_bbox in zip(bbox_list, bbox_img):
        #         result_dict['img_bbox'] = img_bbox
        return bbox_list


    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        raise NotImplementedError
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        raise NotImplementedError
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats

    def aug_test_pts(self, feats, img_metas, rescale=False):
        raise NotImplementedError
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return merged_bboxes

    def show_results(self, data, result, out_dir, show=False, score_thr=None, gt_bboxes=None):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
            show (bool, optional): Determines whether you are
                going to show result by open3d.
                Defaults to False.
            score_thr (float, optional): Score threshold of bounding boxes.
                Default to None.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'

            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d']
            pred_labels = result[batch_id]['pts_bbox']['labels_3d']


            if score_thr is not None:
                mask = result[batch_id]['pts_bbox']['scores_3d'] > score_thr
                pred_bboxes = pred_bboxes[mask]
                pred_labels = pred_labels[mask]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
                gt_bboxes = Box3DMode.convert(gt_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            gt_bboxes = gt_bboxes.tensor.cpu().numpy()
            show_result(
                points,
                gt_bboxes,
                pred_bboxes,
                out_dir,
                file_name,
                show=show,
                pred_labels=pred_labels)


