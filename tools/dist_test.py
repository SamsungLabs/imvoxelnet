import os

os.system('bash tools/dist_test.sh configs/detr3d/detr3d_nuscenes_multi_view.py work_dirs/detr3d_nuscenes_monocular/latest.pth 2 --eval mAP')
