import os
import mmcv
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser


class_names = (
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
    'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'clothes', 'books',
    'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'night_stand', 'toilet',
    'sink', 'lamp', 'bathtub', 'bag'
)


def run(path):
    data = mmcv.load(path)
    categories = {}
    for category_data in data['categories']:
        if category_data['name'] in class_names:
            categories[category_data['id']] = class_names.index(category_data['name'])
    assert len(categories) == len(class_names)
    annotations = defaultdict(list)
    for annotation_data in data['annotations']:
        annotations[annotation_data['image_id']].append(annotation_data)
    info = []
    for image_data in data['images']:
        r = np.array(image_data['rot_mat'])
        yaw = np.arctan(-r[2][0] / r[0][0])
        pitch = np.arctan(r[1][0] / np.sqrt(r[0][0] ** 2 + r[2][0] ** 2))
        roll = np.arctan(-r[1][2] / r[1][1])
        assert(np.isclose(yaw, 0))

        # follow Total3DUnderstanding
        t = np.array([[0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
        r = t @ r.T
        # follow DepthInstance3DBoxes
        r = r[:, [2, 0, 1]]
        r[2] *= -1
        # invert SUNRGBDMonocularDataset._get_matrices
        r = r.T
        r[:, 1] = -1 * r[:, 1]
        r[:, [1, 2]] = r[:, [2, 1]]

        info.append({
            'image': {
                'image_path': os.path.join('OFFICIAL_SUNRGBD', image_data['file_name'])
            },
            'calib': {
                # invert SUNRGBDMonocularDataset._get_matrices
                'K': np.array(image_data['K']).T,
                'Rt': r
            },
            'layout': np.array([
                image_data['room_center'][2],
                image_data['room_center'][0],
                image_data['room_center'][1],
                image_data['room_size'][2],
                image_data['room_size'][0],
                image_data['room_size'][1],
                -image_data['room_angle']
            ]),
            'angles': np.array([pitch, roll]),
            'annos': {
                'class': [],
                'gt_boxes_upright_depth': []
            }
        })
        for annotation in annotations[image_data['id']]:
            if annotation['category_id'] in categories:
                info[-1]['annos']['class'].append(categories[annotation['category_id']])
                info[-1]['annos']['gt_boxes_upright_depth'].append([
                    annotation['center'][2],
                    annotation['center'][0],
                    annotation['center'][1],
                    annotation['size'][2],
                    annotation['size'][0],
                    annotation['size'][1],
                    -annotation['angle']
                ])
        info[-1]['annos']['class'] = np.array(info[-1]['annos']['class'])
        info[-1]['annos']['gt_boxes_upright_depth'] = np.array(info[-1]['annos']['gt_boxes_upright_depth'])
        info[-1]['annos']['gt_num'] = len(info[-1]['annos']['class'])
    mmcv.dump(info, path.replace('.json', '.pkl'), 'pkl')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/sunrgbd')
    args = parser.parse_args()

    run(os.path.join(args.path, 'sunrgbd_total_infos_train.json'))
    run(os.path.join(args.path, 'sunrgbd_total_infos_val.json'))
