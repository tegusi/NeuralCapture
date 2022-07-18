import os

import numpy as np
import torch
import torch.nn.functional as F
import imageio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import src.utils.geometry as geometry
import src.utils.image as image_util
import src.utils.render as render_util
from src.datasets.dataset import Dataset
from src.models.mnerf import MNeRF, Embedding
from src.models.light import MLPLight
from src.models.smpl import SMPL
from src.models.render import render
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
    model, model_fine, model_bg, model_emb = [DDP(x, [args.local_rank]) for x in [model, model_fine, model_bg, model_emb]]
    return model, model_fine, model_bg, smpl_wrapper, model_emb


def train():

    args = get_configs()
    node_count = int(os.environ['WORLD_SIZE'])
    node_rank = int(os.environ['NODE_RANK'])
    gpus_per_node = torch.cuda.device_count()
    gpu_rank = int(os.environ['LOCAL_RANK'])
    global_rank = node_rank * gpus_per_node + gpu_rank
    args.world_size = int(os.environ['WORLD_SIZE'])
    print(f"world_size{args.world_size}")
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.global_rank = global_rank
    total_gpu = (node_rank + 1) * node_count
    print(f"local_rank{args.local_rank}")
    print(f"global_rank{args.global_rank}")
    dist.init_process_group(
        backend='nccl',
    )
    torch.cuda.set_device(args.local_rank)
    scale = 1 / args.scale

    # create log dir and copy the config file
    output_dir = args.output_dir
    if args.global_rank == 0:
        if not os.path.exists(output_dir) and args.global_rank == 0:
            os.makedirs(output_dir)
        f = os.path.join(output_dir, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        sum_logger = SummaryWriter(output_dir) if args.global_rank == 0 else None

    # prepare data
    dataset = Dataset(args.data_dir)
    test_dataset = Dataset(args.test_dir)
    train_frames = dataset.train_frames
    test_frames = test_dataset.test_frames
    print(f"length{len(test_frames) * args.global_rank // total_gpu}")

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

    # set up optimizer
    params = []
    for net in [model, model_fine, model_bg, model_emb]:
        if net is not None:
            for k, v in net.named_parameters():
                if net is model_emb and 'body_poses' in k:
                    continue
                # for different lr settings
                params += [{"params": [v], "lr":args.lrate}]

    def lr_scheduler(epoch):
        return 0.5 ** (epoch / args.lrate_decay)

    params_emb = []
    for k, v in model_emb.named_parameters():
        if 'body_poses' in k:
            params_emb += [{"params": [v], "lr":args.lrate * 10}]
    
    optimizer_smpl = torch.optim.Adam(params_emb)
    optimizer = torch.optim.Adam(params)
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer_smpl, lr_lambda=lr_scheduler)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)

    # network parameters
    kwargs_train = {
        'model': model,
        'model_fine': model_fine,
        'model_bg': model_bg,
        'model_emb': model_emb,
        'N_samples': args.N_samples,
        'N_importance': args.N_importance,
        'training': True,
        'stratified': True,
    }

    kwargs_test = {
        k: kwargs_train[k] for k in kwargs_train}
    kwargs_test['training'] = False
    kwargs_test['stratified'] = False

    # Prepare raybatch tensor if batching random rays
    N_iters = 400000

    np.random.seed(args.global_rank)
    for i in range(1, N_iters+1):
        idx = np.random.randint(len(train_frames))
        i_frame = train_frames[idx]

        # camera parameters
        intrinsic, extrinsic = dataset.camera(i_frame, scale)

        # body parameters
        shape = model_emb.module.shape(idx)
        pose = model_emb.module.pose(idx)
        trans = model_emb.module.trans(idx)
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

        # get keypoint silhouette
        keypoints_2d = render_util.projection(keypoints, extrinsic, intrinsic)

        # sample camera rays
        if np.random.rand() > 0.3:
            bbox2d = geometry.bbox_2d(keypoints_2d[:, :2], expand=1.1)
            coords = render_util.sampling_coords(
                img_W, img_H, args.train_rays, mode='crop', bbox=bbox2d)
        else:
            coords = render_util.sampling_coords(
                img_W, img_H, args.train_rays, mode='full')
        rays = geometry.camera_rays(intrinsic, img_W, img_H, extrinsic, graphics_coordinate=False, center=True)
        batch_rays = rays[coords[:, 0], coords[:, 1]]
        target_c = target[coords[:, 0], coords[:, 1]]

        # render
        ret = render(intrinsic, extrinsic, keypoints, pose = pose, shape = shape, trans = trans, idx = idx, rays=batch_rays,
                     test_rays=args.test_rays, **kwargs_train)

        # loss and optimize
        rgb = ret['rgb_map']

        loss = 0
        img_loss = F.mse_loss(rgb, target_c)
        loss += img_loss

        psnr = image_util.psnr(img_loss)
        if args.optimize_smpl:
            pose_loss = torch.norm((model_emb.module.body_poses.weight - ori_poses)[idx]) ** 2
            loss += 2 * pose_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.optimize_smpl and i % int(len(train_frames) * 3) == 0:
            optimizer_smpl.step()
            optimizer_smpl.zero_grad()
        scheduler1.step()
        scheduler2.step()

        if args.global_rank == 0:
            print(f"Iter:{i} Loss:{loss.item()} PSNR: {psnr.item()}")

            # save checkpoints
            if i % args.i_weights == 0:
                path = os.path.join(output_dir, '{:06d}.tar'.format(i))
                if kwargs_train['model_bg'] is not None:
                    torch.save({
                        'global_step': i,
                        'model': kwargs_train['model'].state_dict(),
                        'model_fine': kwargs_train['model_fine'].state_dict(),
                        'model_bg': kwargs_train['model_bg'].state_dict(),
                        'model_emb': kwargs_train['model_emb'].state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, path)
                else:
                    torch.save({
                        'global_step': i,
                        'model': kwargs_train['model'].state_dict(),
                        'model_fine': kwargs_train['model_fine'].state_dict(),
                        'model_emb': kwargs_train['model_emb'].state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, path)
                print('Saved checkpoints at', path)
            
            # save results
            if i % args.i_print == 0:
                diff = torch.norm(model_emb.module.body_poses.weight - ori_poses) ** 2
                sum_logger.add_scalar('pose_diff', diff, i)
                sum_logger.add_scalar('pose_loss', pose_loss.item(), i)
                sum_logger.add_scalar('loss', loss.item(), i)
                sum_logger.add_scalar('PSNR', psnr.item(), i)
                sum_logger.add_scalar('img_loss', img_loss.item(), i)
                sum_logger.add_scalar('lr1', scheduler1.get_lr()[0], i)
                sum_logger.add_scalar('lr2', scheduler2.get_lr()[0], i)

            # save images
        if i % args.i_img == 0:
            test_intrinsic, test_extrinsic = test_dataset.camera(0, scale)
            local_H, local_W, _ = test_dataset.image(0, scale).shape

            with torch.no_grad():
                if model_bg is not None:
                    _cam_rays = geometry.camera_rays(
                        test_intrinsic, local_W, local_H, test_extrinsic, False, True)
                    _bg = batchify(model_bg, 16, _cam_rays[..., 3:6])[..., :3]
                    _bg = torch.sigmoid(_bg)*(1+2*eps) - eps
                    _bg = torch.clamp(_bg, 0., 1.)
                    _bg = image_util.whc2cwh(_bg).cpu()
                else:
                    _bg = torch.zeros(3, local_H, local_W).cpu()

                mse_test = 0
                psnr_test = 0
                ssim_test = 0
                rgbs = []

                split_test = test_frames[len(test_frames) * args.global_rank // total_gpu: len(test_frames) * (args.global_rank + 1) // total_gpu]
                if len(split_test) > 0:
                    i_log = np.random.choice(split_test)
                for i_test in split_test:
                    shape = test_dataset.shape(i_test)
                    pose = test_dataset.pose(i_test)
                    trans = test_dataset.trans(i_test)
                    shape = shape.unsqueeze(0)
                    pose = pose.unsqueeze(0)
                    trans = trans.unsqueeze(0)
                    keypoints_r = (smpl_wrapper_test(pose, shape, trans)[0])[0]
                    gt = test_dataset.image(i_test,scale)
                    if not args.use_bkgd:
                        gt = gt * test_dataset.mask(i_test, scale).unsqueeze(-1)
                    gt = image_util.whc2cwh(gt).cpu()

                    ret = render(
                        test_intrinsic, test_extrinsic, keypoints_r, rays=None, pose=pose, shape=shape, trans=trans,
                        test_rays=args.test_rays, W=local_W, H=local_H, **kwargs_test)
                    rgb = image_util.whc2cwh(ret['rgb_map']).cpu()
                    rgb = torch.clamp(rgb, 0., 1.)
                    alpha = image_util.whc2cwh(ret['alpha_map']).cpu()
                    rgb_clean = rgb * alpha
                    rgb = rgb * alpha + (1-alpha)*_bg
                    rgb = torch.clamp(rgb, 0., 1.)

                    mse_ = F.mse_loss(rgb, gt)
                    mse_test += mse_
                    psnr_ = image_util.psnr(mse_)
                    psnr_test += psnr_
                    ssim_ = image_util.ssim(rgb, gt)
                    ssim_test += ssim_
                    rgbs.append(image_util.cwh2whc(rgb))
                    os.makedirs(os.path.join(output_dir, '{:06d}'.format(i)), exist_ok=True)
                    imageio.imwrite(os.path.join(output_dir, '{:06d}'.format(i), '{:06d}.jpg'.format(i_test)), image_util.cwh2whc(rgb))

                    # only save one image
                    if i_test == i_log and args.global_rank==0:
                        # rotation 
                        _rot = torch.randn((1, 3))*np.pi
                        _rot_mat = geometry.quat2mat(geometry.rodrigues(_rot))[0]
                        center_r = torch.mean(keypoints_r, dim=0, keepdim=True)
                        keypoints_r = torch.matmul(
                            keypoints_r-center_r, _rot_mat) + center_r

                        ret = render(
                            test_intrinsic, test_extrinsic, keypoints_r, rays=None, pose=pose, shape=shape, trans=trans, rot=_rot,
                            test_rays=args.test_rays, W=local_W, H=local_H, **kwargs_test)
                        rgb_r = image_util.whc2cwh(ret['rgb_map']).cpu()
                        rgb_r = torch.clamp(rgb_r, 0., 1.)
                        alpha_r = image_util.whc2cwh(ret['alpha_map']).cpu()
                        rgb_r = rgb_r * alpha_r + (1-alpha_r)*_bg
                        rgb_r = torch.clamp(rgb_r, 0., 1.)

                        sum_logger.add_image('rgb_r', rgb_r, i)
                        sum_logger.add_image('alpha_r', alpha_r, i)
                        sum_logger.add_image('rgb', rgb, i)
                        sum_logger.add_image('rgb_clean', rgb_clean, i)
                        sum_logger.add_image('alpha', alpha, i)

                        sum_logger.add_image('gt', gt, i)
                        sum_logger.add_image('bg', _bg, i)

                # imageio.mimwrite(os.path.join(output_dir, '{:06d}.gif'.format(i)), rgbs, fps=5)
                dist.barrier()
                dist.all_reduce(mse_test, dist.ReduceOp.SUM)
                dist.all_reduce(psnr_test, dist.ReduceOp.SUM)
                dist.all_reduce(ssim_test, dist.ReduceOp.SUM)
                if args.global_rank==0:
                    sum_logger.add_scalar('mse', mse_test / len(test_frames), i)
                    sum_logger.add_scalar('psnr', psnr_test / len(test_frames), i)
                    sum_logger.add_scalar('ssim', ssim_test / len(test_frames), i)

if __name__ == '__main__':
    train()
