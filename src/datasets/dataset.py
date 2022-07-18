import os
import glob

import cv2
import imageio
import scipy.io
import src.utils.geometry as geometry
import numpy as np
import torch

# AMASS dataset.
# https://amass.is.tue.mpg.de/
SMPL_IDX = [6866, 701, 5680, 1746, 6700, 1456, 2544, 6290, 3787, 5153, 716,
            6472, 4057, 5450, 1529, 3111, 5340, 3492, 6426, 4748, 3463, 926,
            3310, 3725, 6818, 5049, 6368, 4842, 1929, 6380, 2983, 2920, 2608,
            4669, 210, 4924, 6839, 6516, 3442, 6482, 2950, 690, 720, 446, 3333,
            5798, 1012, 1495, 3566, 4186, 385, 5305, 977, 832, 1561, 2708,
            4991, 1097, 1652, 3005, 3126, 2496, 6005, 5234, 3030, 955, 4614,
            847, 5417, 4434, 2346, 5952, 5090, 3339, 4418, 6193, 2958, 2892,
            4333, 6261, 6307, 2066, 6437, 6475, 3867, 3050, 1546, 640, 985,
            1343, 1158, 90, 5281, 4519, 2800, 268, 1318, 3243, 822, 165, 4932,
            2262, 5156, 3481, 1780, 644, 656, 865, 4467, 1382, 5641, 5670,
            6585, 3675, 1820, 1499, 5346, 5226, 2264, 3202, 4844, 6375, 1060,
            4883, 4720, 3498, 6142, 2659]

class Dataset:
    def __init__(self, data_root):
        self.data_root = data_root
        data = scipy.io.loadmat(os.path.join(data_root, 'camera.mat'))
        if 'camK' in data:
            self.camK = data['camK'].astype(np.float32)
        else:
            _camK = np.eye(3)
            _camK[:2, :2] = np.diag(data['color_focal_length'][0])
            _camK[:2, 2] = data['color_center'][0]
            self.camK = _camK.astype(np.float32)
        self.c2w = data['c2w'].astype(np.float32)

        rgb_frames = glob.glob(os.path.join(data_root, 'rgb/*.png'))
        self.rgb_format = 'rgb_frame_{}.png'
        self.mask_format = 'mask_frame_{}.png'
        self.obj_format = 'smpl_frame_{}.obj'
        self.param_format = 'smpl_frame_{}.npz'
        indices = []
        for fname in rgb_frames:
            fname = os.path.basename(fname)
            fname = os.path.splitext(fname)[0]
            idx = int(fname[10:])
            if os.path.isfile(
                    os.path.join(data_root, 'obj',
                                 self.obj_format.format(idx))):
                indices.append(idx)

        self.indices = sorted(indices)
        if os.path.exists(os.path.join(data_root, 'train_frames.txt')):
            with open(os.path.join(data_root, 'train_frames.txt')) as f:
                self.train_frames = [int(x)-1 for x in f.readlines()]
        else:
            self.train_frames = np.arange(0, len(self.indices))

        if os.path.exists(os.path.join(data_root, 'test_frames.txt')):
            with open(os.path.join(data_root, 'test_frames.txt')) as f:
                self.test_frames = [int(x)-1 for x in f.readlines()]
        else:
            self.test_frames = np.arange(len(self.indices) - 10, len(self.indices))

    @staticmethod
    def load_obj(file_path):
        v = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            elem = line.split()
            if elem[0] == 'v':
                v.append(elem[1:])
        v = np.array(v).astype(np.float32)
        return v

    def camera(self, idx, scale=1.0):
        camK = self.camK*scale
        camK[2, 2] = 1.0
        if self.c2w.ndim == 3:
            if self.camK.ndim == 3:
                camK = self.camK[idx]*scale
                camK[2, 2] = 1.0
                return torch.tensor(camK), torch.tensor(self.c2w[idx])
            else:
                return torch.tensor(camK), torch.tensor(self.c2w[idx])
        else:
            return torch.tensor(camK), torch.tensor(self.c2w)

    def image(self, idx, scale=1.0):
        img_path = os.path.join(self.data_root, 'rgb',
                                self.rgb_format.format(self.indices[idx]))
        img = imageio.imread(img_path)
        img = img.astype(np.float32)/255.0

        if scale < 1.0:
            img = cv2.resize(img, dsize=None,
                             fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # the last channel is useless
        return torch.tensor(img[..., :3])

    def mask(self, idx, scale=1.0):
        img_path = os.path.join(self.data_root, 'mask',
                                self.mask_format.format(self.indices[idx]))
        if os.path.exists(img_path):
            img = imageio.imread(img_path)
        else:
            img = imageio.imread(os.path.join(self.data_root, 'rgb',
                                self.rgb_format.format(self.indices[idx])))
            img = np.ones((img.shape[0], img.shape[1]))

        if scale < 1.0:
            img = cv2.resize(img, dsize=None,
                             fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # the last channel is useless
        img = np.clip(img,0,1)
        return torch.tensor(img).float()

    def shape(self, idx):
        if not os.path.exists(os.path.join(self.data_root, 'param')):
            return None
        obj_path = os.path.join(self.data_root, 'param', self.param_format.format(self.indices[idx]))
        v = np.load(obj_path, encoding = 'latin1', allow_pickle=True)['results'].item()
        return torch.tensor(v['betas']).float()
    
    def pose(self, idx):
        if not os.path.exists(os.path.join(self.data_root, 'param')):
            return None
        obj_path = os.path.join(self.data_root, 'param', self.param_format.format(self.indices[idx]))
        v = np.load(obj_path, encoding = 'latin1', allow_pickle=True)['results'].item()
        return torch.tensor(v['pose']).float()

    def trans(self, idx):
        if not os.path.exists(os.path.join(self.data_root, 'param')):
            return None
        obj_path = os.path.join(self.data_root, 'param', self.param_format.format(self.indices[idx]))
        v = np.load(obj_path, encoding = 'latin1', allow_pickle=True)['results'].item()
        if 'trans' not in v:
            return torch.zeros(3)
        return torch.tensor(v['trans']).float()

    def smpl_v(self, idx):
        obj_path = os.path.join(self.data_root, 'obj',
                                self.obj_format.format(self.indices[idx]))
        v = self.load_obj(obj_path)
        v = torch.tensor(v)

        return v

    def __len__(self):
        return len(self.indices)