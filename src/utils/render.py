import numpy as np
import torch
import torch.nn.functional as F
from . import geometry
from .misc import batchify

# MERL database
# https://www.merl.com/brdf/


def sdf2alpha(sdf):
    # sigmoid(s) s->inf
    return F.relu((sdf[..., 1:]-sdf[..., :-1]/sdf[..., 1:]))


# provide activation to sigma out of this function.
def sigma2alpha(sigma, dists):
    return 1.-torch.exp(-sigma*dists)

# provide this sigma dists explicitly


def sigma2weights(sigma, dists):
    alpha = sigma2alpha(sigma, dists)
    weights = alpha \
        * torch.cumprod(
            torch.cat([torch.ones([alpha.shape[0], 1]), 1.-alpha + 1e-10], -1),
            dim=-1)[:, :-1]
    return weights


def sampling_coords(W, H, train_rays, mode='full', mask=None, bbox=None):
    if mode == 'full':
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, H-1, H),
                torch.linspace(0, W-1, W)),
            dim=-1)
    elif mode == 'crop':
        bbox = [int(x) for x in bbox]
        x = np.clip(bbox[0], 0, W-1-1)
        y = np.clip(bbox[1], 0, H-1-1)
        w = np.clip(bbox[2], 1, W-1-x)
        h = np.clip(bbox[3], 1, H-1-y)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(y, y+h-1, h),
                torch.linspace(x, x+w-1, w)),
            dim=-1)
    elif mode == 'mask':
        xx, yy = torch.where(mask)
        coords = torch.stack([xx, yy], dim=-1).T
    else:
        raise ValueError('Unsupported sampling mode: {}'.format(mode))
    coords = torch.reshape(coords, [-1, 2])
    if train_rays > coords.shape[0]:
        select_inds = np.random.choice(
            coords.shape[0], size=[train_rays], replace=True)
    else:
        select_inds = np.random.choice(
            coords.shape[0], size=[train_rays], replace=False)
    select_coords = coords[select_inds].long()
    return select_coords

# Hierarchical sampling (section 5.2)
def importance_sampling(bins, weights, N_samples, uniform=False, eps=1e-7):
    # Get pdf
    weights = weights + eps  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if uniform:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples




eps = 1e-2

def render_image(camK, c2w, keypoints, rays=None, test_rays=1024, W=None, H=None, **kwargs):
    if rays is None:
        rays = geometry.camera_rays(camK, W, H, c2w, False, True)
    sh = rays.shape

    # Create ray batch
    rays = rays.reshape(-1, 6)

    all_ret = batchify(render_rays, test_rays, rays,
                       keypoints=keypoints, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret


def render_rays(ray_batch,
                model,
                model_fine,
                model_bg,
                model_emb,
                pose,
                shape, 
                keypoints,
                N_samples,
                N_importance,
                trans=None,
                idx=None,
                rot=None,
                training=False,
                stratified=False,
                sigma_noise_std=0.):

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]

    # only sample nearby points
    bounds = geometry.ray_mesh_intersection(ray_batch, keypoints.detach(), 0.04)
    near = bounds[..., 0]
    far = bounds[..., 1]
    near[near < 0.] = 0.

    valid_id = torch.where((far-near) > 1e-6)[0]

    rgb_map = torch.zeros((N_rays, 3))
    acc_map = torch.zeros((N_rays, 1))

    if model_bg is None:
        ray_bg = torch.zeros((N_rays, 4))
    else:
        ray_bg = batchify(model_bg, 1024, rays_d)

    # no valid rays
    if len(valid_id) == 0:
        ret = {}
        ret['rgb_map'] = torch.sigmoid(ray_bg[..., :3])*(1+2*eps) - eps
        ret['alpha_map'] = acc_map
        return ret

    # sample valid points
    if training:
        nearest = torch.min(near[valid_id])
        farest = torch.max(far[valid_id])
        near[...] = nearest
        far[...] = farest
        valid_id = torch.where(far > 0)[0]
    invalid_id = torch.where((far-near) <= 1e-6)[0]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    t_vals = t_vals.expand([N_rays, N_samples])

    if stratified:
        mids = .5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(t_vals.shape)
        t_vals = lower + (upper - lower) * t_rand

    z_vals = near[valid_id].unsqueeze(-1)*(1.-t_vals[valid_id, ...]) \
        + far[valid_id].unsqueeze(-1)*t_vals[valid_id, ...]

    bg_valid = ray_bg[valid_id, :]
    o_valid = rays_o[valid_id, :]
    d_valid = rays_d[valid_id, :]

    z_mid = 0.5*(z_vals[..., 1:] + z_vals[..., :-1])
    pts = o_valid[..., None, :] + d_valid[..., None, :]*z_mid[..., :, None]

    # query embedding
    emb = model_emb(torch.cat([pts, d_valid[:,None,...].expand_as(pts)], -1), pose, shape, trans, rot, keypoints, idx)
    # rendering
    raw = batchify(model, N_samples, emb)

    sigma = raw['sigma'][..., 0]
    sigma = torch.cat([sigma, bg_valid[:, None, -1]], dim=-1)
    sigma = sigma + torch.randn(sigma.shape)*sigma_noise_std

    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, dists[:, -1:]], dim=-1)
    dists = dists * torch.norm(d_valid[..., None, :], dim=-1)
    weights = sigma2weights(F.softplus(sigma-1), dists)

    rgb = torch.cat([raw['rgb'], bg_valid[:, None, :3]], dim=-2)
    rgb = torch.sigmoid(rgb)*(1+2*eps) - eps
    rgb_valid = torch.sum(rgb*weights[..., None], dim=-2)
    weights = weights[:, :-1]
    acc_valid = torch.sum(weights, -1)

    # importance sampling
    if N_importance > 0:
        t_samples = importance_sampling(
            z_mid, weights[..., 1:], N_importance, uniform=(not stratified))
        t_samples = t_samples.detach()
        z_vals, _ = torch.sort(
            torch.cat([z_vals, t_samples], -1), -1)
        z_mid = (z_vals[..., 1:] + z_vals[..., :-1])/2.
        pts = o_valid[..., None, :] + d_valid[..., None, :]*z_mid[..., :, None]

        emb = model_emb(torch.cat([pts, d_valid[:,None,...].expand_as(pts)], -1), pose, shape, trans, rot, keypoints, idx)
        run_fn = model if model_fine is None else model_fine
        raw = batchify(run_fn, N_samples, emb)

        sigma = raw['sigma'][..., 0]
        sigma = torch.cat([sigma, bg_valid[:, None, -1]], dim=-1)
        sigma = sigma + torch.randn(sigma.shape)*sigma_noise_std

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, dists[:, -1:]], dim=-1)
        dists = dists * torch.norm(d_valid[..., None, :], dim=-1)
        weights = sigma2weights(F.softplus(sigma-1), dists)

        rgb = torch.cat([raw['rgb'], bg_valid[:, None, :3]], dim=-2)
        rgb = torch.sigmoid(rgb)*(1+2*eps) - eps
        rgb_valid = torch.sum(rgb*weights[..., None], dim=-2)
        weights = weights[:, :-1]
        acc_valid = torch.sum(weights, -1)

    rgb_map[valid_id, ...] = rgb_valid
    acc_map[valid_id, ...] = acc_valid.unsqueeze(-1)
    if len(invalid_id) > 0:
        rgb_map[invalid_id] = torch.sigmoid(ray_bg[invalid_id, :3])\
            * (1+2*eps) - eps
    ret = {'rgb_map': rgb_map, 'alpha_map': acc_map}
    return ret