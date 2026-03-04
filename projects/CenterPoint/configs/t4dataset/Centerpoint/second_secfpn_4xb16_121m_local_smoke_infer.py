_base_ = [
    "./second_secfpn_4xb16_121m_j6gen2_base_amp.py",
]

# Local inference config for /home/taiga/ml_lake/t4-dataset smoke split.
data_root = "/home/taiga/ml_lake/t4-dataset/"
info_directory_path = "info/local_smoke/"
dataset_test_groups = dict(_delete_=True, local_smoke="t4dataset_local_smoke_infos_test.pkl")

# Avoid blocking on an unavailable local mlflow server.
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", name="visualizer", vis_backends=vis_backends)

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args=None,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=1,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=None,
        test_mode=True,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=[-122.4, -122.4, -3.0, 122.4, 122.4, 5.0]),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=(
            "timestamp",
            "num_pts_feats",
            "lidar2img",
            "depth2img",
            "cam2img",
            "box_type_3d",
            "sample_idx",
            "sample_token",
            "lidar_path",
            "img_path",
            "ori_cam2img",
            "cam2global",
            "lidar2cam",
            "ego2global",
        ),
    ),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        ann_file=info_directory_path + "t4dataset_local_smoke_infos_test.pkl",
        pipeline=test_pipeline,
    ),
)

test_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + info_directory_path + "t4dataset_local_smoke_infos_test.pkl",
)
