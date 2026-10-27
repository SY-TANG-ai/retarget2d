from functional.motion import trans_motion3d, normalize_motion, get_local3d
import os
from torch.utils.data import Dataset
import torch
import glob
import numpy as np


FLAT_TEST_CHARACTERS = ['Ty', 'Sporty_Granny', 'X_Bot', 'Y_Bot']


def _is_flat_layout(data_dir):
    return bool(glob.glob(os.path.join(data_dir, '*__*', '*.npy')))


def _scan_flat_layout(data_dir):
    entries = {}
    for combo_dir in sorted(glob.glob(os.path.join(data_dir, '*__*'))):
        if not os.path.isdir(combo_dir):
            continue
        combo_name = os.path.basename(combo_dir)
        if '__' not in combo_name:
            continue
        char_name, motion_name = combo_name.split('__', 1)
        for path in sorted(glob.glob(os.path.join(combo_dir, '*.npy'))):
            motion_item = '{}/{}'.format(motion_name, os.path.basename(path))
            entries[(char_name, motion_item)] = path
    return entries


def _split_flat_characters(character_names, phase):
    character_names = sorted(character_names)
    test_names = [name for name in FLAT_TEST_CHARACTERS if name in character_names]
    if not test_names or len(test_names) >= len(character_names):
        nr_test = max(1, len(character_names) // 4)
        test_names = character_names[-nr_test:]
    train_names = [name for name in character_names if name not in test_names]
    return train_names if phase == 'train' else test_names


class _MixamoDatasetBase(Dataset):
    def __init__(self, phase, config):
        super(_MixamoDatasetBase, self).__init__()

        assert phase in ['train', 'test']
        self.data_root = config.data_dir
        self.phase = phase
        self.unit = config.unit
        self.view_angles = config.view_angles
        self.meanpose_path = config.meanpose_path
        self.stdpose_path = config.stdpose_path
        self.flat_layout = _is_flat_layout(config.data_dir)
        self.flat_entries = None

        if self.flat_layout:
            self.flat_entries = _scan_flat_layout(config.data_dir)
            all_chars = sorted({char for char, _ in self.flat_entries.keys()})
            self.character_names = _split_flat_characters(all_chars, phase)
            motion_sets = []
            for char in self.character_names:
                motion_sets.append({motion for c, motion in self.flat_entries.keys() if c == char})
            common_motions = set.intersection(*motion_sets) if motion_sets else set()
            self.motion_names = sorted(common_motions)
        else:
            self.data_root = os.path.join(config.data_dir, phase)
            # FIXME : decide character_names
            if phase == 'train':
                self.character_names = ['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Malcolm',
                                        'Pearl', 'Warrok', 'Globin', 'Kaya', 'PeanutMan']
            else:
                self.character_names = ['Ty', 'Andromeda', 'Pumpkinhulk', 'SportyGranny']

            items = sorted(glob.glob(os.path.join(self.data_root, self.character_names[0], '*/motions/*.npy')))
            char_root = os.path.join(self.data_root, self.character_names[0])
            self.motion_names = [os.path.relpath(x, char_root).replace(os.sep, '/') for x in items]

        self.aug = phase == 'train'
        if not self.character_names:
            raise RuntimeError('No characters found for phase {} under {}'.format(phase, config.data_dir))
        if not self.motion_names:
            raise RuntimeError('No motion windows found for phase {} under {}'.format(phase, config.data_dir))

        self.mean_pose, self.std_pose = get_meanpose(config)

    def build_item(self, mot_name, char_name):
        """
        :param mot_name: animation_name/motions/xxx.npy
        :param char_name: character_name
        :return:
        """
        if self.flat_layout:
            try:
                return self.flat_entries[(char_name, mot_name)]
            except KeyError:
                raise RuntimeError('Missing flat motion item: {} / {}'.format(char_name, mot_name))
        return os.path.join(self.data_root, char_name, mot_name)

    @staticmethod
    def gen_aug_param(rotate=False):
        if rotate:
            return {'ratio': np.random.uniform(0.8, 1.2),
                    'roll': np.random.uniform((-np.pi / 9, -np.pi / 9, -np.pi / 6), (np.pi / 9, np.pi / 9, np.pi / 6))}
        else:
            return {'ratio': np.random.uniform(0.5, 1.5)}

    @staticmethod
    def augmentation(data, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        if param is None:
            return data, param

        # rotate
        if 'roll' in param.keys():
            cx, cy, cz = np.cos(param['roll'])
            sx, sy, sz = np.sin(param['roll'])
            mat33_x = np.array([
                [1, 0, 0],
                [0, cx, -sx],
                [0, sx, cx]
            ], dtype='float')
            mat33_y = np.array([
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy]
            ], dtype='float')
            mat33_z = np.array([
                [cz, -sz, 0],
                [sz, cz, 0],
                [0, 0, 1]
            ], dtype='float')
            data = mat33_x @ mat33_y @ mat33_z @ data

        # scale
        if 'ratio' in param.keys():
            data = data * param['ratio']

        return data, param

    def preprocessing(self, item, view_angle=None, param=None):
        """
        :param item: filename built from self.build_tiem
        :return:
        """
        motion3d = np.load(item)

        if self.aug:
            motion3d, param = self.augmentation(motion3d, param)

        # convert 3d to 2d
        local3d = None
        if view_angle is not None:
            local3d = get_local3d(motion3d, view_angle)

        motion_proj = trans_motion3d(motion3d, local3d, self.unit)
        motion_proj = normalize_motion(motion_proj, self.mean_pose, self.std_pose)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))   # reshape to (joints*2, len_frames)
        motion_proj = torch.Tensor(motion_proj) # FIXME : change to torch.from_numpy?
        return motion_proj

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.motion_names) * len(self.character_names)


def get_meanpose(config):
    meanpose_path = config.meanpose_path
    stdpose_path = config.stdpose_path
    if os.path.exists(meanpose_path) and os.path.exists(stdpose_path):
        meanpose = np.load(meanpose_path)
        stdpose = np.load(stdpose_path)
    else:
        meanpose, stdpose = gen_meanpose(config)
        np.save(meanpose_path, meanpose)
        np.save(stdpose_path, stdpose)
        print("meanpose saved at {}".format(meanpose_path))
        print("stdpose saved at {}".format(stdpose_path))
    return meanpose, stdpose


def gen_meanpose(config):
    if _is_flat_layout(config.data_dir):
        entries = _scan_flat_layout(config.data_dir)
        train_chars = _split_flat_characters({char for char, _ in entries.keys()}, 'train')
        all_paths = [path for (char, _), path in entries.items() if char in train_chars]
    else:
        all_paths = glob.glob(os.path.join(config.data_dir, 'train', '*/*/motions/*.npy'))
    all_joints = []

    for path in all_paths:
        motion3d = np.load(path)
        local3d = None
        if config.view_angles is None:
            motion_proj = trans_motion3d(motion3d, local3d)
            all_joints.append(motion_proj)
        else:
            for angle in config.view_angles:
                local3d = get_local3d(motion3d, angle)
                motion_proj = trans_motion3d(motion3d.copy(), local3d)
                all_joints.append(motion_proj)

    all_joints = np.concatenate(all_joints, axis=2)

    meanpose = np.mean(all_joints, axis=2)
    stdpose = np.std(all_joints, axis=2)
    stdpose[np.where(stdpose == 0)] = 1e-9
    return meanpose, stdpose
