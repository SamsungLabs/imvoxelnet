import os
import skimage.io
import skimage.draw
import numpy as np

from .builder import DATASETS


@DATASETS.register_module()
class CBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)


class MultiViewMixin:
    # tab20 cmap from matplotlib
    COLORS = [
        [31, 119, 180],
        [174, 199, 232],
        [256, 127, 14],
        [256, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [256, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229]
    ]

    @staticmethod
    def draw_corners(img, corners, color, projection):
        corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
        corners_2d_3 = corners_3d_4 @ projection.T
        z_mask = corners_2d_3[:, 2] > 0
        corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
        corners_2d = corners_2d.astype(np.int)
        for i, j in [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]:
            if z_mask[i] and z_mask[j]:
                ci = corners_2d[i]
                cj = corners_2d[j]
                rr, cc, val = skimage.draw.line_aa(ci[1], ci[0], cj[1], cj[0])
                mask = np.logical_and.reduce((
                    rr >= 0,
                    rr < img.shape[0],
                    cc >= 0,
                    cc < img.shape[1]
                ))
                val = np.broadcast_to(val[mask][:, np.newaxis], (len(val[mask]), 3))
                img[rr[mask], cc[mask], :] = val * color

    def show(self, results, out_dir):
        assert out_dir is not None, 'Expect out_dir, got none.'
        for i, result in enumerate(results):
            info = self.get_data_info(i)
            img = skimage.io.imread(info['img_info'][0]['filename'])
            extrinsic = info['lidar2img']['extrinsic'][0]
            intrinsic = info['lidar2img']['intrinsic'][:3, :3]
            projection = intrinsic @ extrinsic[:3]
            corners = result['boxes_3d'].corners.numpy()
            scores = result['scores_3d'].numpy()
            labels = result['labels_3d'].numpy()
            for corner, score, label in zip(corners, scores, labels):
                self.draw_corners(img, corner, self.COLORS[label], projection)
        skimage.io.imsave(os.path.join(out_dir, 'tmp.png'), img)
