import os

os.system('bash tools/dist_test.sh configs/atlas/atlas_scannet.py work_dirs/atlas_scannet/latest.pth 2 --eval mAP')
