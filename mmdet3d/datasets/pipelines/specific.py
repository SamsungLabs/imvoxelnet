import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


@PIPELINES.register_module()
class NuScenesMultiViewPipeline:
    def __init__(self, transforms):
        self.transforms = Compose(transforms)

    def __call__(self, results):
        aug_data = []
        for i in range(len(results['img_info'])):
            _results = dict()
            for key in ['img_prefix', 'img_info', 'lidar2img']:
                _results[key] = results[key][i]
            for key in ['box_type_3d', 'box_mode_3d']:
                _results[key] = results[key]
            aug_data.append(self.transforms(_results))
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict


@PIPELINES.register_module()
class ScanNetMultiViewPipeline:
    def __init__(self, transforms, n_images):
        self.transforms = Compose(transforms)
        self.n_images = n_images

    def __call__(self, results):
        imgs = []
        lidar2imgs = []
        ids = np.arange(len(results['img_info']))
        replace = True if self.n_images > len(ids) else False
        ids = np.random.choice(ids, self.n_images, replace=replace)
        for i in ids.tolist():
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            lidar2imgs.append(results['lidar2img'][i])
            if i == 0:
                for key in _results.keys():
                    if key not in ['img', 'lidar2img']:
                        results[key] = _results[key]
        results['img'] = imgs
        results['lidar2img'] = lidar2imgs
        return results
