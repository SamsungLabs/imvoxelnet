import os
import mmcv
import numpy as np
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .nuscenes_converter import get_available_scenes, NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode


def create_nuscenes_monocular_infos(root_path,
                                    info_prefix,
                                    version='v1.0-trainval'):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    train_scenes = splits.train
    val_scenes = splits.val

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])

    print('n train scenes: {}, n val scenes: {}'.format(
        len(train_scenes), len(val_scenes)))
    train_monocular_infos, val_monocular_infos, train_multi_view_infos, val_multi_view_infos = \
        _fill_trainval_infos(nusc, train_scenes)
    print('n monocular train samples: {}, n monocular val samples: {}'.format(
        len(train_monocular_infos), len(val_monocular_infos)))
    mmcv.dump(train_monocular_infos, os.path.join(root_path, '{}_monocular_infos_train.pkl'.format(info_prefix)))
    mmcv.dump(val_monocular_infos, os.path.join(root_path, '{}_monocular_infos_val.pkl'.format(info_prefix)))
    print('n multi-view train samples: {}, n multi-view val samples: {}'.format(
        len(train_multi_view_infos), len(val_multi_view_infos)))
    mmcv.dump(train_multi_view_infos, os.path.join(root_path, '{}_multi_view_infos_train.pkl'.format(info_prefix)))
    mmcv.dump(val_multi_view_infos, os.path.join(root_path, '{}_multi_view_infos_val.pkl'.format(info_prefix)))


def _fill_trainval_infos(nusc, train_scenes):
    train_monocular_infos, val_monocular_infos = [], []
    train_multi_view_infos, val_multi_view_infos = [], []
    class_names = {name: index for index, name in enumerate(NuScenesDataset.CLASSES)}

    for sample in mmcv.track_iter_progress(nusc.sample):
        if sample['scene_token'] in train_scenes:
            monocular_infos = train_monocular_infos
            multi_view_infos = train_multi_view_infos
        else:
            monocular_infos = val_monocular_infos
            multi_view_infos = val_multi_view_infos
        multi_view_infos.append({'token': sample['token'], 'images': []})
        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]

            # adapted from NuScenes.get_sample_data
            sd_record = nusc.get('sample_data', cam_token)
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
            cam_path = sd_record['filename']
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            rotation = Quaternion(cs_record['rotation']).inverse.rotation_matrix @ \
                       Quaternion(pose_record['rotation']).inverse.rotation_matrix @ \
                       np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T  # Lidar -> Depth
            translation = Quaternion(cs_record['rotation']).inverse.rotation_matrix @ \
                          Quaternion(pose_record['rotation']).inverse.rotation_matrix @ \
                          -np.array(pose_record['translation']) + \
                          Quaternion(cs_record['rotation']).inverse.rotation_matrix @ \
                          -np.array(cs_record['translation'])
            width, height = sd_record['width'], sd_record['height']
            boxes, classes = [], []

            for box in nusc.get_boxes(cam_token):
                # we need to convert rotation to SECOND format.
                tmp_box = (box.center.tolist() +
                    box.wlh.tolist() +
                    [-box.orientation.yaw_pitch_roll[0] - np.pi / 2])

                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                center_3d = box.center
                center_2d = view_points(center_3d[..., None], cam_intrinsic, normalize=True)[:2, 0]

                # True if a center is at least 0.1 meter in front of the camera and in image.
                if (0 < center_2d[0] < width and
                    0 < center_2d[1] < height and
                    center_3d[2] > 0.1 and
                    box.name in NuScenesDataset.NameMapping):
                    tmp_box = LiDARInstance3DBoxes(
                        np.array(tmp_box)[None, ...],
                        origin=(0.5, 0.5, 0.5)).convert_to(Box3DMode.DEPTH)
                    # Shift box to remove translation from consideration
                    tmp_box.translate(rotation.T @ translation)
                    # Move origin back to (0.5, 0.5, 0.5)
                    tmp_box = np.concatenate((tmp_box.gravity_center.numpy()[0], tmp_box.tensor.numpy()[0, 3:]))
                    boxes.append(tmp_box)
                    classes.append(class_names[NuScenesDataset.NameMapping[box.name]])

                    # from mmdet3d.core.bbox import DepthInstance3DBoxes
                    # depth_box = DepthInstance3DBoxes(tmp_box[None, ...], origin=(.5, .5, .5))
                    # corners_3d = depth_box.corners.numpy()[0]
                    # corners_3d = (rotation @ corners_3d.T).T
                    # corners_2d_3 = np.dot(cam_intrinsic, corners_3d.T).T
                    # corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
                    #
                    # print('corners_2d:', corners_2d)
                    # print('view_points corners 2d:', view_points(box.corners(), cam_intrinsic, normalize=True)[:2])
                    # raise

            monocular_info = {
                'annos': {
                    'gt_boxes_upright_depth': np.array(boxes),
                    'class': np.array(classes),
                    'gt_num': len(classes)
                },
                'image': {
                    'image_path': cam_path,
                },
                'calib': {
                    'intrinsic': cam_intrinsic,
                    'extrinsic': rotation.T
                }
            }
            multi_view_infos[-1]['images'].append({
                'image': monocular_info['image'],
                'calib': {
                    'intrinsic': monocular_info['calib']['intrinsic'],
                    'extrinsic': monocular_info['calib']['extrinsic'],
                    'global': rotation.T @ translation
                },
            })
            monocular_infos.append(monocular_info)

    return train_monocular_infos, val_monocular_infos, train_multi_view_infos, val_multi_view_infos
