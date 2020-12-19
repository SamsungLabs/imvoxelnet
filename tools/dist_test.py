import os

os.system('bash tools/dist_test.sh configs/detr3d/detr3d_sunrgbd-3d-10class.py work_dirs/detr3d_sunrgbd-3d-10class/latest.pth 2 --eval mAP')
