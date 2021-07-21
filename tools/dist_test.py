import os

os.system('bash tools/dist_test.sh configs/imvoxelnet/imvoxelnet_kitti.py work_dirs/imvoxelnet_kitti/20210503_214214.pth 2 --eval mAP')
