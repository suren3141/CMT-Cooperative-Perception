from os import path as osp

import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import auto_fp16

from mmdet3d.core import Box3DMode, Coord3DMode, show_result
from mmdet3d.models.detectors.base import Base3DDetector

class BaseCoop3DDetector(Base3DDetector):
    """Base class for ccoperative detectors."""

    @auto_fp16(apply_to=("vehicle_img", "vehicle_points", "infrastructure_img", "infrastructure_points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def show_results(self, data, result, out_dir, show=False, score_thr=None):
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

            pred_bboxes = result[batch_id]['boxes_3d']
            pred_labels = result[batch_id]['labels_3d']

            if score_thr is not None:
                mask = result[batch_id]['scores_3d'] > score_thr
                pred_bboxes = pred_bboxes[mask]
                pred_labels = pred_labels[mask]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(
                points,
                None,
                pred_bboxes,
                out_dir,
                file_name,
                show=show,
                pred_labels=pred_labels)
