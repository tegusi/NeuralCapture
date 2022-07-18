import os

import numpy as np
import torch
import torch.nn.functional as F
import imageio

import src.utils.geometry as geometry
import src.utils.image as image_util
from src.datasets.dataset import Dataset
from src.models.mnerf import MNeRF, Embedding
from src.models.light import MLPLight
from src.models.smpl import SMPL
from src.utils.render import render_image

from src.utils.config import get_configs
from src.utils.misc import batchify

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
eps = 1e-3

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    if 'male' in args.data_dir:
        smpl_wrapper = SMPL(args.smpl_dir, gender='male')
    elif 'female' in args.data_dir:
        smpl_wrapper = SMPL(args.smpl_dir, gender='female')
    else:
        smpl_wrapper = SMPL(args.smpl_dir, gender='neutral')
    if args.use_direction:
        model = MNeRF(input_dim=args.embed_dim + 27)
        model_fine = MNeRF(input_dim=args.embed_dim + 27)
        model_emb = Embedding(smpl_wrapper, args.body_betas, args.body_poses, args.body_trans, args.latent_dim, output_dim=args.embed_dim, use_dir=True)
    else:
        model = MNeRF(input_dim=args.embed_dim)
        model_fine = MNeRF(input_dim=args.embed_dim)
        model_emb = Embedding(smpl_wrapper, args.body_betas, args.body_poses, args.body_trans, args.latent_dim, output_dim=args.embed_dim, use_dir=False)

    model_bg = MLPLight(out_channels=4)
    
    # Load model
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        model_fine.load_state_dict(ckpt['model_fine'])
        model_emb.load_state_dict(ckpt['model_emb'])
        if 'model_bg' in ckpt:
            model_bg.load_state_dict(ckpt['model_bg'])
        smpl_wrapper.load_state_dict(ckpt['smpl'])
    return model, model_fine, model_bg, smpl_wrapper, model_emb


def test():
    args = get_configs()
    args.n_gpus = torch.cuda.device_count()
    scale = 1 / args.scale

    # create log dir and copy the config file
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare data
    dataset = Dataset(args.data_dir)
    test_dataset = Dataset(args.test_dir)
    train_frames = dataset.train_frames
    test_frames = test_dataset.test_frames

    # create network
    args.body_betas = torch.stack([dataset.shape(i) for i in train_frames])
    args.body_poses = torch.stack([dataset.pose(i) for i in train_frames])
    ori_poses = args.body_poses.detach().clone()
    args.body_trans = torch.stack([dataset.trans(i) for i in train_frames])
    model, model_fine, model_bg, smpl_wrapper, model_emb = create_nerf(args)
    if 'male' in args.test_dir:
        smpl_wrapper_test = SMPL(args.smpl_dir, gender='male')
    elif 'female' in args.test_dir:
        smpl_wrapper_test = SMPL(args.smpl_dir, gender='female')
    else:
        smpl_wrapper_test = SMPL(args.smpl_dir, gender='neutral')
    if not args.use_bkgd:
        model_bg = None

    # network parameters
    kwargs_test = {
        'model': model,
        'model_fine': model_fine,
        'model_bg': model_bg,
        'model_emb': model_emb,
        'N_samples': args.N_samples,
        'N_importance': args.N_importance,
        'training': False,
        'stratified': False,
    }

    rgbs = []
    for i in range(0, len(test_frames)):
        # Random from one frame and one view
        i_frame = test_frames[i]
        # camera parameters
        intrinsic, extrinsic = dataset.camera(i_frame, scale)

        # body parameters
        shape = dataset.shape(i_frame)
        pose = dataset.pose(i_frame)
        trans = dataset.trans(i_frame)
        mask = dataset.mask(i_frame, scale)

        shape = shape.unsqueeze(0)
        pose = pose.unsqueeze(0)
        trans = trans.unsqueeze(0)
        keypoints = smpl_wrapper(pose, shape, trans)[0][0]

        # ground truth image
        target = dataset.image(i_frame, scale)
        if not args.use_bkgd:
            target = target * mask.unsqueeze(-1)
        
        img_H = target.shape[0]
        img_W = target.shape[1]

        with torch.no_grad():
            if model_bg is not None:
                _cam_rays = geometry.camera_rays(
                    intrinsic, img_W, img_H, extrinsic, False, True)
                _bg = batchify(model_bg, 16, _cam_rays[..., 3:6])[..., :3]
                _bg = torch.sigmoid(_bg)*(1+2*eps) - eps
                _bg = torch.clamp(_bg, 0., 1.)
                _bg = image_util.whc2cwh(_bg).cpu()
            else:
                _bg = torch.zeros(3, img_H, img_W).cpu()


            ret = render_image(
                intrinsic, extrinsic, keypoints, rays=None, theta=pose, shape=shape, trans=trans,
                test_rays=args.test_rays, W=img_W, H=img_H, **kwargs_test)
            rgb = ret['rgb_map'].cpu()
            alpha = ret['alpha_map'].cpu()
            rgb = rgb * alpha + (1-alpha)*torch.ones_like(_bg)
            rgb = torch.clamp(rgb, 0., 1.)

            imageio.imwrite(os.path.join(output_dir, f'{i:02d}.png'), rgb)
            imageio.imwrite(os.path.join(output_dir, f'{i:02d}_a.png'), alpha)
            rgbs.append(rgb)
    imageio.mimwrite(os.path.join(output_dir, '{:06d}.gif'.format(i)), rgbs, fps=5)


if __name__ == '__main__':
    test()
