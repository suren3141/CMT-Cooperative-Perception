# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
import os

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import torch
from mmcv import Config
from mmcv.runner import load_state_dict, _load_checkpoint, _load_checkpoint_with_prefix


from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CmtDetector to CmtCoopDetector')
    parser.add_argument('config', help='config file')
    parser.add_argument('--vehicle_checkpoint', help='checkpoint file')
    parser.add_argument('--infrastructure_checkpoint', help='checkpoint file')
    parser.add_argument('--vehicle_lidar', help='checkpoint file')
    parser.add_argument('--vehicle_camera', help='checkpoint file')
    parser.add_argument('--infrastructure_lidar', help='checkpoint file')
    parser.add_argument('--infrastructure_camera', help='checkpoint file')
    parser.add_argument('--out', help='path of the output checkpoint file')
    args = parser.parse_args()
    return args


def parse_config(config, config_strings=None):
    """Parse config from strings.

    Args:
        config_strings (string): strings of model config.

    Returns:
        Config: model config
    """

    cfg = Config.fromfile(config)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    cfg = compat_cfg(cfg)


    return cfg

def update_ckpt(orig_ckpt, INSERT_PREFIX, PERMUTE_PREFIX, DEL_PREFIX):
    
    converted_ckpt = orig_ckpt.copy()

    # # Delete some useless keys
    # for key in DEL_KEYS:
    #     converted_ckpt.pop(key)

    ALL_KEYS = list(converted_ckpt.keys())

    for old_key in ALL_KEYS:
        for k in DEL_PREFIX:
            if old_key.startswith(k):
                converted_ckpt.pop(old_key)


    # Insert prefix to keys
    PREFIX_KEYS = dict()
    for old_key in converted_ckpt.keys():
        for k in INSERT_PREFIX.keys():
            if old_key.startswith(k):
                new_key = old_key.replace(k, INSERT_PREFIX[k])
                PREFIX_KEYS[new_key] = old_key
    for new_key, old_key in PREFIX_KEYS.items():
        converted_ckpt[new_key] = converted_ckpt.pop(old_key)

    # Permutes vals beginning with key
    for old_key in converted_ckpt.keys():
        for k in PERMUTE_PREFIX.keys():
            if old_key.startswith(k) and converted_ckpt[old_key].dim() == len(PERMUTE_PREFIX[k]):
                converted_ckpt[old_key] = torch.permute(converted_ckpt[old_key], PERMUTE_PREFIX[k])
    


    '''
    # Rename keys with specific prefix
    RENAME_KEYS = dict()
    for old_key in converted_ckpt.keys():
        for rename_prefix in RENAME_PREFIX.keys():
            if rename_prefix in old_key:
                new_key = old_key.replace(rename_prefix,
                                          RENAME_PREFIX[rename_prefix])
                RENAME_KEYS[new_key] = old_key
    for new_key, old_key in RENAME_KEYS.items():
        converted_ckpt[new_key] = converted_ckpt.pop(old_key)
    '''

    # Extract weights and rename the keys
    '''
    for new_key, (old_key, indices) in EXTRACT_KEYS.items():
        cur_layers = orig_ckpt[old_key]
        converted_layers = []
        for (start, end) in indices:
            if end != -1:
                converted_layers.append(cur_layers[start:end])
            else:
                converted_layers.append(cur_layers[start:])
        converted_layers = torch.cat(converted_layers, 0)
        converted_ckpt[new_key] = converted_layers
        if old_key in converted_ckpt.keys():
            converted_ckpt.pop(old_key)
    '''

    # named_layers = dict(model.named_modules())
    # matched_layers = []

    # for k in converted_ckpt.keys():
    #     if k in named_layers.keys():
    #         matched_layers.append(k)

    return converted_ckpt



def update_ckpt_vehicle(orig_ckpt, prefix=None, wo_trans=False):

    # if cfg['dataset_type'] == 'ScanNetDataset':
    #     NUM_CLASSES = 18
    # elif cfg['dataset_type'] == 'SUNRGBDDataset':
    #     NUM_CLASSES = 10
    # else:
    #     raise NotImplementedError

    INSERT_PREFIX = {
        'img_backbone' : 'vehicle_model.img_backbone',
        'img_neck' : 'vehicle_model.img_neck',
        'pts_voxel_encoder' : 'vehicle_model.pts_voxel_encoder',
        'pts_middle_encoder' : 'vehicle_model.pts_middle_encoder',
        'pts_backbone' : 'vehicle_model.pts_backbone',
        'pts_neck' : 'vehicle_model.pts_neck'
    }

    if prefix is not None:
        INSERT_PREFIX_KEYS = list(INSERT_PREFIX.keys())
        for k in INSERT_PREFIX_KEYS:
            if not k.startswith(prefix) : INSERT_PREFIX.pop(k)
    
    print(INSERT_PREFIX)

    # mismatch of shapes between pts_middle_encoder in torchsparse v1 and v2
    PERMUTE_PREFIX = {
        'vehicle_model.pts_middle_encoder.conv_input' : (1, 2, 3, 4, 0),
        'vehicle_model.pts_middle_encoder.encoder_layers' : (1, 2, 3, 4, 0),
        'vehicle_model.pts_middle_encoder.conv_out' : (1, 2, 3, 4, 0),
    }

    DEL_PREFIX = {
        'infrastructure_model',         #  Delete infra_model feature extractor
        'pts_bbox_head.task_heads',
    }

    if wo_trans : DEL_PREFIX.add('pts_bbox_head.transformer')


    RENAME_PREFIX = {
        # 'bbox_head.conv_pred.0': 'bbox_head.conv_pred.shared_convs.layer0',
        # 'bbox_head.conv_pred.1': 'bbox_head.conv_pred.shared_convs.layer1'
    }

    DEL_KEYS = [
        # 'bbox_head.conv_pred.0.bn.num_batches_tracked',
        # 'bbox_head.conv_pred.1.bn.num_batches_tracked'
    ]

    EXTRACT_KEYS = {
    #     'bbox_head.conv_pred.conv_cls.weight':
    #     ('bbox_head.conv_pred.conv_out.weight', [(0, 2), (-NUM_CLASSES, -1)]),
    #     'bbox_head.conv_pred.conv_cls.bias':
    #     ('bbox_head.conv_pred.conv_out.bias', [(0, 2), (-NUM_CLASSES, -1)]),
    #     'bbox_head.conv_pred.conv_reg.weight':
    #     ('bbox_head.conv_pred.conv_out.weight', [(2, -NUM_CLASSES)]),
    #     'bbox_head.conv_pred.conv_reg.bias':
    #     ('bbox_head.conv_pred.conv_out.bias', [(2, -NUM_CLASSES)])
    }

    converted_ckpt = update_ckpt(orig_ckpt, INSERT_PREFIX, PERMUTE_PREFIX, DEL_PREFIX)


    return converted_ckpt



def update_ckpt_infrastructure(orig_ckpt, prefix=None, wo_trans=False):


    # if cfg['dataset_type'] == 'ScanNetDataset':
    #     NUM_CLASSES = 18
    # elif cfg['dataset_type'] == 'SUNRGBDDataset':
    #     NUM_CLASSES = 10
    # else:
    #     raise NotImplementedError

    INSERT_PREFIX = {
        'img_backbone' : 'infrastructure_model.img_backbone',
        'img_neck' : 'infrastructure_model.img_neck',
        'pts_voxel_encoder' : 'infrastructure_model.pts_voxel_encoder',
        'pts_middle_encoder' : 'infrastructure_model.pts_middle_encoder',
        'pts_backbone' : 'infrastructure_model.pts_backbone',
        'pts_neck' : 'infrastructure_model.pts_neck'
    }

    if prefix is not None:
        INSERT_PREFIX_KEYS = list(INSERT_PREFIX.keys())
        for k in INSERT_PREFIX_KEYS:
            if not k.startswith(prefix) : INSERT_PREFIX.pop(k)
    
    print(INSERT_PREFIX)


    PERMUTE_PREFIX = {
        'infrastructure_model.pts_middle_encoder.conv_input' : (1, 2, 3, 4, 0),
        'infrastructure_model.pts_middle_encoder.encoder_layers' : (1, 2, 3, 4, 0),
        'infrastructure_model.pts_middle_encoder.conv_out' : (1, 2, 3, 4, 0),
    }

    DEL_PREFIX = {
        'vehicle_model',        # Delete vehicle_model feature extractor
        'pts_bbox_head.task_heads',
    }

    if wo_trans : DEL_PREFIX.add('pts_bbox_head.transformer')

    RENAME_PREFIX = {
        # 'bbox_head.conv_pred.0': 'bbox_head.conv_pred.shared_convs.layer0',
        # 'bbox_head.conv_pred.1': 'bbox_head.conv_pred.shared_convs.layer1'
    }

    DEL_KEYS = [
        # 'bbox_head.conv_pred.0.bn.num_batches_tracked',
        # 'bbox_head.conv_pred.1.bn.num_batches_tracked'
    ]

    EXTRACT_KEYS = {
    #     'bbox_head.conv_pred.conv_cls.weight':
    #     ('bbox_head.conv_pred.conv_out.weight', [(0, 2), (-NUM_CLASSES, -1)]),
    #     'bbox_head.conv_pred.conv_cls.bias':
    #     ('bbox_head.conv_pred.conv_out.bias', [(0, 2), (-NUM_CLASSES, -1)]),
    #     'bbox_head.conv_pred.conv_reg.weight':
    #     ('bbox_head.conv_pred.conv_out.weight', [(2, -NUM_CLASSES)]),
    #     'bbox_head.conv_pred.conv_reg.bias':
    #     ('bbox_head.conv_pred.conv_out.bias', [(2, -NUM_CLASSES)])
    }

    converted_ckpt = update_ckpt(orig_ckpt, INSERT_PREFIX, PERMUTE_PREFIX, DEL_PREFIX)


    return converted_ckpt


def main():
    """Convert keys in checkpoints from CmtDetector to CmtCoopDetector.

    """
    args = parse_args()
    # checkpoint = torch.load(args.checkpoint)
    # cfg = parse_config(checkpoint['meta']['config'])
    cfg = parse_config(args.config)

    # Build the model and load checkpoint
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    if args.vehicle_checkpoint is not None:
        vehicle_checkpoint = _load_checkpoint(filename=args.vehicle_checkpoint)
        if 'state_dict' in vehicle_checkpoint:
            vehicle_checkpoint = vehicle_checkpoint['state_dict']
        vehicle_ckpt = update_ckpt_vehicle(vehicle_checkpoint)
    else:
        if args.vehicle_lidar is not None:
            vehicle_lidar = _load_checkpoint(filename=args.vehicle_lidar)
            if 'state_dict' in vehicle_lidar:
                vehicle_lidar = vehicle_lidar['state_dict']
            vehicle_lidar = update_ckpt_vehicle(vehicle_lidar, prefix='pts')
        else:
            vehicle_lidar = {}
        if args.vehicle_camera is not None:
            vehicle_camera = _load_checkpoint(filename=args.vehicle_camera)
            if 'state_dict' in vehicle_camera:
                vehicle_camera = vehicle_camera['state_dict']
            vehicle_camera = update_ckpt_vehicle(vehicle_camera, prefix='img')
        else:
            vehicle_camera = {}

        shared_keys = [k for k in vehicle_camera.keys() if k in vehicle_lidar.keys()]
        if len(shared_keys) : print("Shared keys : ", shared_keys)

        # vehicle_ckpt = {**vehicle_lidar, **vehicle_camera}
        vehicle_ckpt = {**vehicle_camera, **vehicle_lidar}

    if args.infrastructure_checkpoint is not None:
        infrastructure_checkpoint = _load_checkpoint(filename=args.infrastructure_checkpoint)
        if 'state_dict' in infrastructure_checkpoint:
            infrastructure_checkpoint = infrastructure_checkpoint['state_dict']
        infrastructure_ckpt = update_ckpt_infrastructure(infrastructure_checkpoint)
    else:
        if args.infrastructure_lidar is not None:
            infrastructure_lidar = _load_checkpoint(filename=args.infrastructure_lidar)
            if 'state_dict' in infrastructure_lidar:
                infrastructure_lidar = infrastructure_lidar['state_dict']
            infrastructure_lidar = update_ckpt_infrastructure(infrastructure_lidar, prefix='pts')
        else:
            infrastructure_lidar = {}
        if args.infrastructure_camera is not None:
            infrastructure_camera = _load_checkpoint(filename=args.infrastructure_camera)
            if 'state_dict' in infrastructure_camera:
                infrastructure_camera = infrastructure_camera['state_dict']
            infrastructure_camera = update_ckpt_infrastructure(infrastructure_camera, prefix='img')
        else:
            infrastructure_camera = {}

        # infrastructure_ckpt = {**infrastructure_lidar, **infrastructure_camera}
        infrastructure_ckpt = {**infrastructure_camera, **infrastructure_lidar}


    converted_ckpt = {**vehicle_ckpt, **infrastructure_ckpt}

    if 'state_dict' in converted_ckpt:
        checkpoint = converted_ckpt['state_dict']
    else:
        checkpoint = converted_ckpt


    # Check the converted checkpoint by loading to the model
    load_state_dict(model, checkpoint, strict=False)

    out_path = os.path.split(args.out)[0]
    if not os.path.exists(out_path): os.makedirs(out_path)

    torch.save(checkpoint, args.out)


if __name__ == '__main__':
    main()
