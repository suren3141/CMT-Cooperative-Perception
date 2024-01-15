plugin=True
plugin_dir='projects/mmdet3d_plugin/'

class_names = ['CAR', 'TRAILER', 'TRUCK', 'VAN', 'PEDESTRIAN', 'BUS', 'BICYCLE']

point_cloud_range = [-72.0, -72.0, -8, 72.0, 72.0, 0]
voxel_size = [0.1, 0.1, 0.2]    # 1440 x 1440 x 40

out_size_factor = 8

dataset_type = 'A9NuscCoopDataset'
data_root = 'data/a9_coop_nusc/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
    
ida_aug_conf = {
        "resize_lim": (0.94, 1.25),
        "final_dim": (640, 1600),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCoop'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='TransformLidar2ImgToInfraCoords'),
    dict(
        type='GlobalRotScaleTransImageCoop',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    # dict(
    #     type='RandomFlip3D',
    #     # type='CustomRandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImageCoop', data_aug_conf = ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImageCoop', **img_norm_cfg),
    dict(type='PadMultiViewImageCoop', size_divisor=32),
    dict(type='DefaultFormatBundle3DCoop', class_names=class_names),
    dict(type='Collect3D', keys=['vehicle_img', 'infrastructure_img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape',
                    'vehicle2infrastructure',
                    'vehicle_lidar2img',
                    'infrastructure_lidar2img',
                    # 'depth2img', 
                    # 'cam2img', 
                    'vehicle_pad_shape', 
                    'infrastructure_pad_shape', 
                    'scale_factor', 'flip', 
                    'pcd_horizontal_flip','pcd_vertical_flip', 
                    'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCoop'),
    dict(type='TransformLidar2ImgToInfraCoords'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTransImageCoop',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            # dict(type='RandomFlip3D'),
            dict(type='ResizeCropFlipImageCoop', data_aug_conf = ida_aug_conf, training=False),
            dict(type='NormalizeMultiviewImageCoop', **img_norm_cfg),
            dict(type='PadMultiViewImageCoop', size_divisor=32),
            dict(
                type='DefaultFormatBundle3DCoop',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['vehicle_img', 'infrastructure_img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 
                            'vehicle2infrastructure',
                            'vehicle_lidar2img',
                            'infrastructure_lidar2img',
                            # 'depth2img', 
                            # 'cam2img', 
                            'vehicle_pad_shape', 
                            'infrastructure_pad_shape', 
                            'scale_factor', 'flip', 
                            'pcd_horizontal_flip','pcd_vertical_flip', 
                            'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            # 'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            # 'transformation_3d_flow', 'rot_degree',
                            # 'gt_bboxes_3d', 'gt_labels_3d'
                            ))
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/a9_nusc_coop_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/a9_nusc_coop_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/a9_nusc_coop_infos_test.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

headless_model = dict(
    type='CmtDetector',
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
)

model = dict(
    type='CmtCoopDetector',
    vehicle_model = headless_model,
    infrastructure_model = headless_model,
    coop_fusion_neck = None,
    pts_bbox_head=dict(
        type='CmtImageHeadCoop',
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=len(class_names), class_names=class_names),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0], # TODO : Change 
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names)), 
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmtImageTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),

                    feedforward_channels=1024, #unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
        )))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'infrastructure_model.img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'infrastructure_model.img_neck': dict(lr_mult=0.1),
            'vehicle_model.img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'vehicle_model.img_neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    grad_clip=dict(max_norm=35, norm_type=2),
    )

lr_config = dict(
    policy='cyclic',
    target_ratio=(2, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
evaluation = dict(interval=1)
total_epochs = 20
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=5,
    )
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'pretrained/cmt_a9_debug/camera/coop/IC_cmt_camera_vov_1600x640_cbgs_a9coop_infrastructure_loss_norm_pretrained_debug/VC_cmt_camera_vov_1600x640_cbgs_a9coop_vehicle_loss_norm_pretrained_debug/cmt_camera_vov_1600x640_cbgs_a9coop_loss_norm_debug_pretrained'
load_from = 'pretrained/cmt_a9_debug/camera/coop/IC_cmt_camera_vov_1600x640_cbgs_a9coop_infrastructure_loss_norm_pretrained_debug/VC_cmt_camera_vov_1600x640_cbgs_a9coop_vehicle_loss_norm_pretrained_debug/cmt_camera_vov_1600x640_cbgs_a9coop_loss_norm_debug-pretrained.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
