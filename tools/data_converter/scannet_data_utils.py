import os
import mmcv
import numpy as np
from concurrent import futures as futures
from os import path as osp


class ScanNetData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.classes = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'scannetv2_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = mmcv.list_from_file(split_file)

    def __len__(self):
        return len(self.sample_id_list)

    def get_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_train_instance_data',
                            f'{idx}_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir,
                                    'scannet_train_instance_data',
                                    f'{sample_idx}_vert.npy')
            pts_instance_mask_path = osp.join(self.root_dir,
                                              'scannet_train_instance_data',
                                              f'{sample_idx}_ins_label.npy')
            pts_semantic_mask_path = osp.join(self.root_dir,
                                              'scannet_train_instance_data',
                                              f'{sample_idx}_sem_label.npy')

            points = np.load(pts_filename)
            pts_instance_mask = np.load(pts_instance_mask_path).astype(np.long)
            pts_semantic_mask = np.load(pts_semantic_mask_path).astype(np.long)

            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'instance_mask'))
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'semantic_mask'))

            points.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx}.bin'))
            pts_instance_mask.tofile(
                osp.join(self.root_dir, 'instance_mask', f'{sample_idx}.bin'))
            pts_semantic_mask.tofile(
                osp.join(self.root_dir, 'semantic_mask', f'{sample_idx}.bin'))

            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')
            info['pts_instance_mask_path'] = osp.join('instance_mask',
                                                      f'{sample_idx}.bin')
            info['pts_semantic_mask_path'] = osp.join('semantic_mask',
                                                      f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                boxes_with_classes = self.get_box_label(
                    sample_idx)  # k, 6 + class
                annotations['gt_num'] = boxes_with_classes.shape[0]
                if annotations['gt_num'] != 0:
                    minmax_boxes3d = boxes_with_classes[:, :-1]  # k, 6
                    classes = boxes_with_classes[:, -1]  # k, 1
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    annotations['location'] = minmax_boxes3d[:, :3]
                    annotations['dimensions'] = minmax_boxes3d[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = minmax_boxes3d
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


class ScanNetMonocularData(ScanNetData):
    def process_single_scene(self, sample_idx, has_label):
        info = dict(image_paths=[], pose=[])

        with open(os.path.join(self.root_dir, 'txts', f'{sample_idx}.txt')) as file:
            for line in file.readlines():
                splits = line.split(' = ')
                if splits[0] == 'axisAlignment':
                    axis_align_matrix = np.fromstring(splits[1], sep=' ').reshape(4, 4)
                    break
        info['axis_align_matrix'] = axis_align_matrix

        frame_sub_path = f'sens_reader_100/scans/{sample_idx}/out'
        frame_path = osp.join(self.root_dir, frame_sub_path)
        base_file_names = {x.split('.')[0] for x in os.listdir(frame_path)}
        base_file_names.remove('_info')
        for base_file_name in base_file_names:
            pose = np.loadtxt(osp.join(frame_path, f'{base_file_name}.pose.txt'))
            if np.all(np.isfinite(pose)):
                info['image_paths'].append(osp.join(frame_sub_path, f'{base_file_name}.color.jpg'))
                info['pose'].append(pose)

        with open(osp.join(frame_path, '_info.txt')) as file:
            splits = file.readlines()[7].split(' = ')
            assert splits[0] == 'm_calibrationColorIntrinsic'
            info['intrinsic'] = np.fromstring(splits[1], sep=' ').reshape(4, 4)

        if has_label:
            annotations = {}
            bbox_path = osp.join(self.root_dir, 'mmdetection3d', f'{sample_idx}_bbox.npy')
            boxes_with_classes = np.load(bbox_path)
            annotations['gt_num'] = boxes_with_classes.shape[0]
            if annotations['gt_num'] != 0:
                minmax_boxes3d = boxes_with_classes[:, :-1]  # k, 6
                classes = boxes_with_classes[:, -1]  # k, 1
                annotations['gt_boxes_upright_depth'] = minmax_boxes3d
                annotations['class'] = np.array([
                    self.cat_ids2class[classes[i]]
                    for i in range(annotations['gt_num'])
                ])
            info['annos'] = annotations
        return info

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        infos = []
        for i, sample_idx in enumerate(sample_id_list):
            print(f'{self.split} sample_idx: {sample_idx} {i}/{len(sample_id_list)}')
            infos.append(self.process_single_scene(sample_idx, has_label))
        return infos
