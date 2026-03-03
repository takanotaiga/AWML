_base_ = ["./base.py"]

# Local smoke-test split for quick end-to-end validation on a tiny subset.
info_train_file_name = "t4dataset_local_smoke_infos_train.pkl"
info_val_file_name = "t4dataset_local_smoke_infos_val.pkl"
info_test_file_name = "t4dataset_local_smoke_infos_test.pkl"

info_train_statistics_file_name = "t4dataset_local_smoke_statistics_train.parquet"
info_val_statistics_file_name = "t4dataset_local_smoke_statistics_val.parquet"
info_test_statistics_file_name = "t4dataset_local_smoke_statistics_test.parquet"

dataset_version_list = ["db_j6_v1"]
