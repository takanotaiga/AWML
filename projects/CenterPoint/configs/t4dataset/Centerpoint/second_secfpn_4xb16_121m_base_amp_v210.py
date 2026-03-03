auto_scale_lr = dict(base_batch_size=64, enable=False)
backend_args = None
camera_panels = [
    'data/CAM_FRONT_LEFT',
    'data/CAM_FRONT',
    'data/CAM_FRONT_RIGHT',
    'data/CAM_BACK_LEFT',
    'data/CAM_BACK',
    'data/CAM_BACK_RIGHT',
]
camera_types = {
    'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_FRONT'
}
class_colors = dict(
    barrier=(
        0,
        0,
        0,
    ),
    bicycle=(
        255,
        0,
        30,
    ),
    bus=(
        111,
        255,
        111,
    ),
    car=(
        30,
        144,
        255,
    ),
    construction_vehicle=(
        255,
        255,
        0,
    ),
    motorcycle=(
        100,
        0,
        30,
    ),
    pedestrian=(
        255,
        200,
        200,
    ),
    traffic_cone=(
        120,
        120,
        120,
    ),
    trailer=(
        0,
        255,
        255,
    ),
    truck=(
        140,
        0,
        255,
    ))
class_names = [
    'car',
    'truck',
    'bus',
    'bicycle',
    'pedestrian',
]
clip_grad = dict(max_norm=15, norm_type=2)
custom_hooks = [
    dict(type='MomentumInfoHook'),
    dict(type='LossScaleInfoHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.CenterPoint.models',
        'autoware_ml.detection3d.datasets.t4dataset',
        'autoware_ml.detection3d.evaluation.t4metric.t4metric',
        'autoware_ml.detection3d.datasets.transforms',
        'autoware_ml.hooks',
    ])
data_prefix = dict(
    CAM_BACK='',
    CAM_BACK_LEFT='',
    CAM_BACK_RIGHT='',
    CAM_FRONT='',
    CAM_FRONT_LEFT='',
    CAM_FRONT_RIGHT='',
    pts='',
    sweeps='')
data_root = 'data/t4dataset/'
dataset_test_groups = dict(
    db_base='t4dataset_base_infos_test.pkl',
    db_j6='t4dataset_x2_infos_test.pkl',
    db_j6gen2='t4dataset_j6gen2_infos_test.pkl',
    db_jpntaxi='t4dataset_xx1_infos_test.pkl',
    db_largebus='t4dataset_largebus_infos_test.pkl')
dataset_type = 'T4Dataset'
dataset_version_config_root = 'autoware_ml/configs/t4dataset/'
dataset_version_list = [
    'db_j6gen2_v1',
    'db_j6gen2_v2',
    'db_j6gen2_v4',
    'db_largebus_v1',
    'db_jpntaxi_v1',
    'db_jpntaxi_v2',
    'db_jpntaxi_v4',
    'db_gsm8_v1',
    'db_j6_v1',
    'db_j6_v2',
    'db_j6_v3',
    'db_j6_v5',
]
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='NuScenes metric/T4Metric/mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl', timeout=7200),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_class_range = dict(
    bicycle=121, bus=121, car=121, pedestrian=121, truck=121)
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=1,
        type='LoadPointsFromMultiSweeps',
        use_dim=[
            0,
            1,
            2,
            4,
        ]),
    dict(
        point_cloud_range=[
            -121.6,
            -121.6,
            -3.0,
            121.6,
            121.6,
            5.0,
        ],
        type='PointsRangeFilter'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
filter_attributes = [
    (
        'vehicle.bicycle',
        'vehicle_state.parked',
    ),
    (
        'vehicle.bicycle',
        'cycle_state.without_rider',
    ),
    (
        'vehicle.bicycle',
        'motorcycle_state.without_rider',
    ),
    (
        'vehicle.motorcycle',
        'vehicle_state.parked',
    ),
    (
        'vehicle.motorcycle',
        'cycle_state.without_rider',
    ),
    (
        'vehicle.motorcycle',
        'motorcycle_state.without_rider',
    ),
    (
        'bicycle',
        'vehicle_state.parked',
    ),
    (
        'bicycle',
        'cycle_state.without_rider',
    ),
    (
        'bicycle',
        'motorcycle_state.without_rider',
    ),
    (
        'motorcycle',
        'vehicle_state.parked',
    ),
    (
        'motorcycle',
        'cycle_state.without_rider',
    ),
    (
        'motorcycle',
        'motorcycle_state.without_rider',
    ),
]
grid_size = [
    760,
    760,
    1,
]
info_directory_path = 'info/kokseang_2_1/'
info_test_file_name = 't4dataset_base_infos_test.pkl'
info_train_file_name = 't4dataset_base_infos_train.pkl'
info_val_file_name = 't4dataset_base_infos_val.pkl'
input_modality = dict(
    use_camera=False,
    use_external=False,
    use_lidar=True,
    use_map=False,
    use_radar=False)
launcher = 'none'
lidar_sweep_dims = [
    0,
    1,
    2,
    4,
]
load_from = 'work_dirs/centerpoint_2_1/T4Dataset/second_secfpn_4xb16_121m_base_amp/epoch_49.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
logger_interval = 50
lr = 0.0003
max_epochs = 50
merge_objects = [
    (
        'truck',
        [
            'truck',
            'trailer',
        ],
    ),
]
merge_type = 'extend_longer'
metainfo = dict(classes=[
    'car',
    'truck',
    'bus',
    'bicycle',
    'pedestrian',
])
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            deterministic=True,
            max_num_points=32,
            max_voxels=(
                64000,
                64000,
            ),
            point_cloud_range=[
                -121.6,
                -121.6,
                -3.0,
                121.6,
                121.6,
                5.0,
            ],
            voxel_size=[
                0.32,
                0.32,
                8.0,
            ])),
    pts_backbone=dict(
        conv_cfg=dict(bias=False, type='Conv2d'),
        in_channels=32,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            1,
            2,
            2,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    pts_bbox_head=dict(
        bbox_coder=dict(
            code_size=9,
            max_num=500,
            out_size_factor=1,
            pc_range=[
                -121.6,
                -121.6,
                -3.0,
                121.6,
                121.6,
                5.0,
            ],
            post_center_range=[
                -200.0,
                -200.0,
                -10.0,
                200.0,
                200.0,
                10.0,
            ],
            score_threshold=0.1,
            type='CenterPointBBoxCoder',
            voxel_size=[
                0.32,
                0.32,
                8.0,
            ]),
        common_heads=dict(
            dim=(
                3,
                2,
            ),
            height=(
                1,
                2,
            ),
            reg=(
                2,
                2,
            ),
            rot=(
                2,
                2,
            ),
            vel=(
                2,
                2,
            )),
        in_channels=384,
        loss_bbox=dict(
            loss_weight=0.25, reduction='mean', type='mmdet.L1Loss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='none',
            type='mmdet.AmpGaussianFocalLoss'),
        norm_bbox=True,
        separate_head=dict(
            final_kernel=1, init_bias=-4.595, type='CustomSeparateHead'),
        share_conv_channel=64,
        tasks=[
            dict(
                class_names=[
                    'car',
                    'truck',
                    'bus',
                    'bicycle',
                    'pedestrian',
                ],
                num_class=5),
        ],
        type='CenterHead'),
    pts_middle_encoder=dict(
        in_channels=32, output_shape=(
            760,
            760,
        ), type='PointPillarsScatter'),
    pts_neck=dict(
        in_channels=[
            64,
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            128,
            128,
            128,
        ],
        type='SECONDFPN',
        upsample_cfg=dict(bias=False, type='deconv'),
        upsample_strides=[
            1,
            2,
            4,
        ],
        use_conv_for_no_stride=True),
    pts_voxel_encoder=dict(
        feat_channels=[
            32,
            32,
        ],
        in_channels=4,
        legacy=False,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        point_cloud_range=[
            -121.6,
            -121.6,
            -3.0,
            121.6,
            121.6,
            5.0,
        ],
        type='PillarFeatureNet',
        voxel_size=[
            0.32,
            0.32,
            8.0,
        ],
        with_cluster_center=True,
        with_distance=False,
        with_voxel_center=True),
    test_cfg=dict(
        pts=dict(
            grid_size=[
                760,
                760,
                1,
            ],
            min_radius=[
                1.0,
            ],
            nms_type='circle',
            out_size_factor=1,
            pc_range=[
                -121.6,
                -121.6,
                -3.0,
                121.6,
                121.6,
                5.0,
            ],
            post_center_limit_range=[
                -200.0,
                -200.0,
                -10.0,
                200.0,
                200.0,
                10.0,
            ],
            post_max_size=100,
            voxel_size=[
                0.32,
                0.32,
                8.0,
            ])),
    train_cfg=dict(
        pts=dict(
            code_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
            ],
            dense_reg=1,
            gaussian_overlap=0.1,
            grid_size=[
                760,
                760,
                1,
            ],
            max_objs=500,
            min_radius=2,
            out_size_factor=1,
            point_cloud_range=[
                -121.6,
                -121.6,
                -3.0,
                121.6,
                121.6,
                5.0,
            ],
            voxel_size=[
                0.32,
                0.32,
                8.0,
            ])),
    type='CenterPoint')
name_mapping = dict({
    'ambulance': 'car',
    'animal': 'animal',
    'bicycle': 'bicycle',
    'bus': 'bus',
    'car': 'car',
    'construction_vehicle': 'truck',
    'construction_worker': 'pedestrian',
    'fire_truck': 'truck',
    'forklift': 'car',
    'kart': 'car',
    'motorcycle': 'bicycle',
    'movable_object.barrier': 'barrier',
    'movable_object.debris': 'debris',
    'movable_object.pushable_pullable': 'pushable_pullable',
    'movable_object.traffic_cone': 'traffic_cone',
    'movable_object.trafficcone': 'traffic_cone',
    'pedestrian': 'pedestrian',
    'pedestrian.adult': 'pedestrian',
    'pedestrian.child': 'pedestrian',
    'pedestrian.construction_worker': 'pedestrian',
    'pedestrian.personal_mobility': 'pedestrian',
    'pedestrian.police_officer': 'pedestrian',
    'pedestrian.stroller': 'pedestrian',
    'pedestrian.wheelchair': 'pedestrian',
    'personal_mobility': 'pedestrian',
    'police_car': 'car',
    'police_officer': 'pedestrian',
    'semi_trailer': 'trailer',
    'static_object.bicycle rack': 'bicycle rack',
    'static_object.bicycle_rack': 'bicycle_rack',
    'static_object.bollard': 'bollard',
    'stroller': 'pedestrian',
    'tractor_unit': 'truck',
    'trailer': 'trailer',
    'truck': 'truck',
    'vehicle.ambulance': 'car',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus': 'bus',
    'vehicle.bus (bendy & rigid)': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'truck',
    'vehicle.emergency (ambulance & police)': 'car',
    'vehicle.fire': 'truck',
    'vehicle.motorcycle': 'bicycle',
    'vehicle.police': 'car',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
    'wheelchair': 'pedestrian'
})
num_class = 5
num_workers = 32
optim_wrapper = dict(
    clip_grad=dict(max_norm=15, norm_type=2),
    dtype='float16',
    loss_scale=dict(growth_interval=2000, init_scale=256.0),
    optimizer=dict(lr=0.0003, type='AdamW', weight_decay=0.01),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.0003, type='AdamW', weight_decay=0.01)
out_size_factor = 1
param_scheduler = [
    dict(
        T_max=15,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=15,
        eta_min=0.0029999999999999996,
        type='CosineAnnealingLR'),
    dict(
        T_max=35,
        begin=15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=50,
        eta_min=3e-08,
        type='CosineAnnealingLR'),
    dict(
        T_max=15,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=15,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=35,
        begin=15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=50,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]
point_cloud_range = [
    -121.6,
    -121.6,
    -3.0,
    121.6,
    121.6,
    5.0,
]
point_load_dim = 5
point_use_dim = 3
randomness = dict(deterministic=True, diff_rank_seed=False, seed=0)
resume = False
sweeps_num = 1
sync_bn = 'torch'
test_batch_size = 2
test_cfg = dict()
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='info/kokseang_2_1/t4dataset_base_infos_test.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        class_names=[
            'car',
            'truck',
            'bus',
            'bicycle',
            'pedestrian',
        ],
        data_prefix=dict(
            CAM_BACK='',
            CAM_BACK_LEFT='',
            CAM_BACK_RIGHT='',
            CAM_FRONT='',
            CAM_FRONT_LEFT='',
            CAM_FRONT_RIGHT='',
            pts='',
            sweeps=''),
        data_root='data/t4dataset/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'bus',
            'bicycle',
            'pedestrian',
        ]),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=1,
                type='LoadPointsFromMultiSweeps',
                use_dim=[
                    0,
                    1,
                    2,
                    4,
                ]),
            dict(
                point_cloud_range=[
                    -121.6,
                    -121.6,
                    -3.0,
                    121.6,
                    121.6,
                    5.0,
                ],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='T4Dataset'),
    num_workers=32,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/t4dataset/info/kokseang_2_1/t4dataset_base_infos_test.pkl',
    backend_args=None,
    class_names=[
        'car',
        'truck',
        'bus',
        'bicycle',
        'pedestrian',
    ],
    data_root='data/t4dataset/',
    dataset_name='db_base',
    eval_class_range=dict(
        bicycle=121, bus=121, car=121, pedestrian=121, truck=121),
    filter_attributes=[
        (
            'vehicle.bicycle',
            'vehicle_state.parked',
        ),
        (
            'vehicle.bicycle',
            'cycle_state.without_rider',
        ),
        (
            'vehicle.bicycle',
            'motorcycle_state.without_rider',
        ),
        (
            'vehicle.motorcycle',
            'vehicle_state.parked',
        ),
        (
            'vehicle.motorcycle',
            'cycle_state.without_rider',
        ),
        (
            'vehicle.motorcycle',
            'motorcycle_state.without_rider',
        ),
        (
            'bicycle',
            'vehicle_state.parked',
        ),
        (
            'bicycle',
            'cycle_state.without_rider',
        ),
        (
            'bicycle',
            'motorcycle_state.without_rider',
        ),
        (
            'motorcycle',
            'vehicle_state.parked',
        ),
        (
            'motorcycle',
            'cycle_state.without_rider',
        ),
        (
            'motorcycle',
            'motorcycle_state.without_rider',
        ),
    ],
    metric='bbox',
    name_mapping=dict({
        'ambulance': 'car',
        'animal': 'animal',
        'bicycle': 'bicycle',
        'bus': 'bus',
        'car': 'car',
        'construction_vehicle': 'truck',
        'construction_worker': 'pedestrian',
        'fire_truck': 'truck',
        'forklift': 'car',
        'kart': 'car',
        'motorcycle': 'bicycle',
        'movable_object.barrier': 'barrier',
        'movable_object.debris': 'debris',
        'movable_object.pushable_pullable': 'pushable_pullable',
        'movable_object.traffic_cone': 'traffic_cone',
        'movable_object.trafficcone': 'traffic_cone',
        'pedestrian': 'pedestrian',
        'pedestrian.adult': 'pedestrian',
        'pedestrian.child': 'pedestrian',
        'pedestrian.construction_worker': 'pedestrian',
        'pedestrian.personal_mobility': 'pedestrian',
        'pedestrian.police_officer': 'pedestrian',
        'pedestrian.stroller': 'pedestrian',
        'pedestrian.wheelchair': 'pedestrian',
        'personal_mobility': 'pedestrian',
        'police_car': 'car',
        'police_officer': 'pedestrian',
        'semi_trailer': 'trailer',
        'static_object.bicycle rack': 'bicycle rack',
        'static_object.bicycle_rack': 'bicycle_rack',
        'static_object.bollard': 'bollard',
        'stroller': 'pedestrian',
        'tractor_unit': 'truck',
        'trailer': 'trailer',
        'truck': 'truck',
        'vehicle.ambulance': 'car',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus': 'bus',
        'vehicle.bus (bendy & rigid)': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'truck',
        'vehicle.emergency (ambulance & police)': 'car',
        'vehicle.fire': 'truck',
        'vehicle.motorcycle': 'bicycle',
        'vehicle.police': 'car',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
        'wheelchair': 'pedestrian'
    }),
    save_csv=True,
    type='T4Metric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=1,
        type='LoadPointsFromMultiSweeps',
        use_dim=[
            0,
            1,
            2,
            4,
        ]),
    dict(
        point_cloud_range=[
            -121.6,
            -121.6,
            -3.0,
            121.6,
            121.6,
            5.0,
        ],
        type='PointsRangeFilter'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
train_batch_size = 16
train_cfg = dict(
    by_epoch=True,
    dynamic_intervals=[
        (
            45,
            1,
        ),
    ],
    max_epochs=50,
    val_interval=5)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='info/kokseang_2_1/t4dataset_base_infos_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        class_names=[
            'car',
            'truck',
            'bus',
            'bicycle',
            'pedestrian',
        ],
        data_prefix=dict(
            CAM_BACK='',
            CAM_BACK_LEFT='',
            CAM_BACK_RIGHT='',
            CAM_FRONT='',
            CAM_FRONT_LEFT='',
            CAM_FRONT_RIGHT='',
            pts='',
            sweeps=''),
        data_root='data/t4dataset/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'bus',
            'bicycle',
            'pedestrian',
        ]),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=1,
                type='LoadPointsFromMultiSweeps',
                use_dim=[
                    0,
                    1,
                    2,
                    4,
                ]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,
                sync_2d=False,
                type='RandomFlip3D'),
            dict(
                rot_range=[
                    -1.571,
                    1.571,
                ],
                scale_ratio_range=[
                    0.8,
                    1.2,
                ],
                translation_std=[
                    1.0,
                    1.0,
                    0.2,
                ],
                type='GlobalRotScaleTrans'),
            dict(
                point_cloud_range=[
                    -121.6,
                    -121.6,
                    -3.0,
                    121.6,
                    121.6,
                    5.0,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -121.6,
                    -121.6,
                    -3.0,
                    121.6,
                    121.6,
                    5.0,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'bus',
                    'bicycle',
                    'pedestrian',
                ],
                type='ObjectNameFilter'),
            dict(min_num_points=5, type='ObjectMinPointsFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='T4Dataset'),
    num_workers=32,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_gpu_size = 4
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=1,
        type='LoadPointsFromMultiSweeps',
        use_dim=[
            0,
            1,
            2,
            4,
        ]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        rot_range=[
            -1.571,
            1.571,
        ],
        scale_ratio_range=[
            0.8,
            1.2,
        ],
        translation_std=[
            1.0,
            1.0,
            0.2,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        point_cloud_range=[
            -121.6,
            -121.6,
            -3.0,
            121.6,
            121.6,
            5.0,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -121.6,
            -121.6,
            -3.0,
            121.6,
            121.6,
            5.0,
        ],
        type='ObjectRangeFilter'),
    dict(
        classes=[
            'car',
            'truck',
            'bus',
            'bicycle',
            'pedestrian',
        ],
        type='ObjectNameFilter'),
    dict(min_num_points=5, type='ObjectMinPointsFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='info/kokseang_2_1/t4dataset_base_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        class_names=[
            'car',
            'truck',
            'bus',
            'bicycle',
            'pedestrian',
        ],
        data_prefix=dict(
            CAM_BACK='',
            CAM_BACK_LEFT='',
            CAM_BACK_RIGHT='',
            CAM_FRONT='',
            CAM_FRONT_LEFT='',
            CAM_FRONT_RIGHT='',
            pts='',
            sweeps=''),
        data_root='data/t4dataset/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'bus',
            'bicycle',
            'pedestrian',
        ]),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=1,
                type='LoadPointsFromMultiSweeps',
                use_dim=[
                    0,
                    1,
                    2,
                    4,
                ]),
            dict(
                point_cloud_range=[
                    -121.6,
                    -121.6,
                    -3.0,
                    121.6,
                    121.6,
                    5.0,
                ],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='T4Dataset'),
    num_workers=32,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/t4dataset/info/kokseang_2_1/t4dataset_base_infos_val.pkl',
    backend_args=None,
    class_names=[
        'car',
        'truck',
        'bus',
        'bicycle',
        'pedestrian',
    ],
    data_root='data/t4dataset/',
    eval_class_range=dict(
        bicycle=121, bus=121, car=121, pedestrian=121, truck=121),
    filter_attributes=[
        (
            'vehicle.bicycle',
            'vehicle_state.parked',
        ),
        (
            'vehicle.bicycle',
            'cycle_state.without_rider',
        ),
        (
            'vehicle.bicycle',
            'motorcycle_state.without_rider',
        ),
        (
            'vehicle.motorcycle',
            'vehicle_state.parked',
        ),
        (
            'vehicle.motorcycle',
            'cycle_state.without_rider',
        ),
        (
            'vehicle.motorcycle',
            'motorcycle_state.without_rider',
        ),
        (
            'bicycle',
            'vehicle_state.parked',
        ),
        (
            'bicycle',
            'cycle_state.without_rider',
        ),
        (
            'bicycle',
            'motorcycle_state.without_rider',
        ),
        (
            'motorcycle',
            'vehicle_state.parked',
        ),
        (
            'motorcycle',
            'cycle_state.without_rider',
        ),
        (
            'motorcycle',
            'motorcycle_state.without_rider',
        ),
    ],
    metric='bbox',
    name_mapping=dict({
        'ambulance': 'car',
        'animal': 'animal',
        'bicycle': 'bicycle',
        'bus': 'bus',
        'car': 'car',
        'construction_vehicle': 'truck',
        'construction_worker': 'pedestrian',
        'fire_truck': 'truck',
        'forklift': 'car',
        'kart': 'car',
        'motorcycle': 'bicycle',
        'movable_object.barrier': 'barrier',
        'movable_object.debris': 'debris',
        'movable_object.pushable_pullable': 'pushable_pullable',
        'movable_object.traffic_cone': 'traffic_cone',
        'movable_object.trafficcone': 'traffic_cone',
        'pedestrian': 'pedestrian',
        'pedestrian.adult': 'pedestrian',
        'pedestrian.child': 'pedestrian',
        'pedestrian.construction_worker': 'pedestrian',
        'pedestrian.personal_mobility': 'pedestrian',
        'pedestrian.police_officer': 'pedestrian',
        'pedestrian.stroller': 'pedestrian',
        'pedestrian.wheelchair': 'pedestrian',
        'personal_mobility': 'pedestrian',
        'police_car': 'car',
        'police_officer': 'pedestrian',
        'semi_trailer': 'trailer',
        'static_object.bicycle rack': 'bicycle rack',
        'static_object.bicycle_rack': 'bicycle_rack',
        'static_object.bollard': 'bollard',
        'stroller': 'pedestrian',
        'tractor_unit': 'truck',
        'trailer': 'trailer',
        'truck': 'truck',
        'vehicle.ambulance': 'car',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus': 'bus',
        'vehicle.bus (bendy & rigid)': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'truck',
        'vehicle.emergency (ambulance & police)': 'car',
        'vehicle.fire': 'truck',
        'vehicle.motorcycle': 'bicycle',
        'vehicle.police': 'car',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
        'wheelchair': 'pedestrian'
    }),
    type='T4Metric')
val_interval = 5
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
voxel_size = [
    0.32,
    0.32,
    8.0,
]
work_dir = 'work_dirs/centerpoint_2_1/T4Dataset/second_secfpn_4xb16_121m_base_amp/'
