'''
Basic geometry operations in torch
'''
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F


def projection(keypoints, c2w, cam):
    """
    Project original 3D keypoints to 2D planes.
    Args:
        keypoints: [N, 3]
        c2w: [4, 4]
        cam: [3, 3]
    Returns:
        keypoints_2d: [N, 2]
    """
    keypoints_2d = torch.matmul(keypoints - c2w[:3, 3], c2w[:3, :3])
    keypoints_2d = torch.matmul(keypoints_2d, cam.T)
    keypoints_2d = keypoints_2d / keypoints_2d[:, 2:]
    return keypoints_2d[:, :2]

def rigid_align(src,dest):
    '''
    calculate the rigid transform between src and tar
    '''

    c_src = torch.mean(src,dim=0)[None,:]
    c_dest = torch.mean(dest,dim=0)[None,:]
    H_mat = torch.mm((src - c_src).transpose(1,0),(dest-c_dest))
    U,S,V = torch.svd(H_mat)
    R = torch.mm(V,U.transpose(1,0))
    if torch.det(R) < 0:
        V[:,2] *= -1
        R = torch.mm(V,U.transpose(1,0))

    #print(R)

    t = - torch.mm(R,c_src.transpose(1,0)) + c_dest.transpose(1,0)
    return R,t
def bbox_2d(v, expand=1.):
    v_min = torch.min(v, dim=0)[0]
    v_max = torch.max(v, dim=0)[0]
    box_size = (v_max - v_min)*expand
    bbox = [v_min[0], v_min[1], box_size[0], box_size[1]]
    return bbox


def bbox_3d(v, expand=1.):
    v_min = torch.min(v, dim=0)[0]
    v_max = torch.max(v, dim=0)[0]

    box_center = (v_min + v_max)/2.0
    box_size = (v_max - v_min)*expand

    box_center = box_center.expand([3, 3])

    box_axis = torch.diag(box_size)

    face_center = box_center - torch.roll(box_axis/2.0, shifts=1, dims=0)

    bbox = torch.cat([face_center, box_axis], dim=0)
    return bbox


def hemispherical_phong_sampling(
        N_sample: int,
        p: torch.Tensor,
        eps: float = 0.) -> torch.Tensor:
    """perform phong(cosine-weighted) sampling on hemisphere.

    Args:
        N_sample: number of samples.
        p: Tensors `p in [0,inf)` is hardness. `0` will result in 
        uniform sampling..
    Returns:
        The sampled theta,phi shaped p.shape + [N_sample]
    """
    u = torch.rand(list(p.shape)+[N_sample])
    v = torch.rand(list(p.shape)+[N_sample])
    u = torch.pow(1-u, 1./(1.+p[..., None]))
    theta = torch.arccos(u*(1-eps))
    phi = 2 * np.pi * v
    return torch.stack([theta, phi], dim=-1)


def hemispherical_uniform_sampling(N_sample) -> torch.Tensor:
    u = torch.rand(N_sample)
    v = torch.rand(N_sample)
    theta = torch.arccos(u)
    phi = 2 * np.pi * v
    return torch.stack([theta, phi], dim=-1)


def spherical_uniform_sampling(N_sample: int) -> torch.Tensor:
    u = torch.rand(N_sample)
    v = torch.rand(N_sample)
    theta = torch.arccos(2*u-1)
    phi = 2 * np.pi * v
    return torch.stack([theta, phi], dim=-1)


def spherical_latlong_grid(height, width):

    theta_step = np.pi/height

    theta = torch.linspace(theta_step, np.pi-theta_step, height)
    phi = torch.linspace(0, 2*np.pi, width)

    phi, theta = torch.meshgrid(phi, theta)
    theta = theta.t()
    phi = phi.t()
    omega = spherical_to_cartesian(torch.stack([theta, phi], dim=-1))

    sin_colat = torch.sin(theta)
    areas = 4 * np.pi * sin_colat / torch.sum(sin_colat)
    return omega, areas


def spherical_to_cartesian(input: torch.Tensor) -> torch.Tensor:
    theta, phi = input[..., 0], input[..., 1]
    x = torch.sin(theta)*torch.sin(phi)
    y = torch.sin(theta)*torch.cos(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


def cartesian_to_spherical(
        xyz: torch.Tensor,
        eps: float = 1e-12) -> torch.Tensor:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    theta = torch.acos(z/(torch.norm(xyz, p=2, dim=-1)+eps))
    phi = torch.atan2(y, x)
    return torch.stack([theta, phi], dim=-1)


def cross_product_matrix(v):
    """skew symmetric form of cross-product matrix

    Args:
        v: tensor of shape `[...,3]`
    Returns:
        The skew symmetric form `[...,3,3]`
    """

    v0 = v[..., 0]
    v1 = v[..., 1]
    v2 = v[..., 2]
    zero = torch.zeros_like(v0)
    mat = torch.stack([
        zero, -v2, v1,
        v2, zero, -v0,
        -v1, v0, zero], dim=-1).view(list(v0.shape)+[3, 3])
    return mat


def reflect(v: torch.Tensor, axis: torch.Tensor):
    """reflect vector a w.r.t. axis

    Args:
        `v`: tensor of shape `[...,3]`.
        `axis`: tensor with the same shape or dim as `a`.
    Returns:
        the reflected vector
    """
    axis = torch.broadcast_to(axis, v.shape)
    h_vec = 2*axis * torch.sum(axis*v, dim=-1, keepdim=True)
    return h_vec - v


def vrrot(a: torch.Tensor, b: torch.Tensor):
    """calculate the rotation matrix that rotates vector a to b.

    Args:
        `a`: tensor of shape `[...,3]`.
        `b`: tensor with the same shape as `a`. `{a,b}` cannot be zero 
        vectors
    Returns:
        The rotation matrix of shape `[...,3,3]` that rotates 
        `a` to `b`.
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    v = torch.cross(a, b, dim=-1)
    c = torch.sum(a*b, dim=-1)
    v_x = cross_product_matrix(v)

    rot = v_x + torch.bmm(v_x, v_x) / (1+c[..., None, None]) \
        + torch.eye(3)
    rot[c <= -1.0, :, :] = -torch.eye(3)
    return rot


def vertices_normals(v, f, vbyf=None):
    if v.dim() > 2:
        raise NotImplementedError('only supports batch-size==1')
    if vbyf is None:
        vbyf = vertex_by_face(f)
    u = v[f[:, 1], :] - v[f[:, 0], :]
    v = v[f[:, 2], :] - v[f[:, 0], :]
    f_normal = torch.cross(u, v)
    f_normal = F.normalize(f_normal, dim=-1)
    v_normal = torch.matmul(vbyf, f_normal)
    v_normal = F.normalize(v_normal, dim=-1)
    return v_normal


def vertex_by_face(f):
    vid = f.T.flatten()
    v_num = torch.max(vid) + 1
    fid = torch.arange(f.shape[0])
    fid = fid.repeat(3)
    val = torch.ones(len(fid))
    index = torch.stack([vid, fid], dim=-1).T
    vbyf = torch.sparse_coo_tensor(
        index, val, size=(v_num, f.shape[0]))
    return vbyf


def lbs(v, w, theta, rig, rig_tree, inverse=False):
    """linear blend skinning

    Args:
        v: vertices of shape `[...,V,3]`
        w: blend weights of shape `[V,J]`
        theta: joint rotations of shape `[...,J,3]`
        rig: rigs used for lbs. `[J,3]`
        rig_tree: rig hierarchy, `[J,1], parent id(i)=rig_tree[i]`
        inverse: whether its inverse blend or forward blend  
    Returns:
        The blended vertices of shape `[...,V,3]`
    """
    n_V, n_J = w.shape
    n_J
    batch_size = theta.shape[:-2]
    Rs = quat2mat(rodrigues(theta))

    _, Ts_out = kinematric_transform(Rs, rig, rig_tree)
    if inverse:
        Ts_out = Ts_out.inverse()

    Ts_vec = Ts_out.reshape(batch_size+(n_J, 16))

    T_w = torch.matmul(Ts_vec.transpose(-1, -2), w.T).transpose(-1, -2)
    T_w = T_w.view(batch_size+(n_V, 4, 4))

    verts = torch.cat([v, torch.ones(
        list(v.shape[:-1])+[1], dtype=v.dtype, device=v.device)], dim=-1)

    verts = torch.matmul(T_w, torch.unsqueeze(verts, -1))
    verts = verts[:, :, :3, 0]
    return verts


def kinematric_transform(Rs, rig, rig_tree):
    """kinematric_transform

    Args:
        Rs: rotation matrix of each joint [...,J,3,3]
        rig: rigs used for lbs. `[J,3]`
        rig_tree: rig hierarchy, `[J,1], parent id(i)=rig_tree[i]`
    Returns:
        The global transform as shape `[...,J,4,4]`
    """
    outT = torch.empty(Rs.shape[:-2]+(4, 4))
    Js = rig.unsqueeze(-1)

    def make_T(R, t):
        Rt = torch.cat([R, t], dim=-1)
        r = torch.zeros(R.shape[:-2]+(1, 4), dtype=R.dtype, device=R.device)
        r[..., 3] = 1
        return torch.cat([Rt, r], dim=-2)

    outT[..., 0, :, :] = make_T(Rs[..., 0, :, :], Js[..., 0, :, :])

    for idj in range(1, len(rig_tree)):
        ipar = rig_tree[idj].long()
        J_local = Js[..., idj, :, :] - Js[..., ipar, :, :]
        T_local = make_T(Rs[..., idj, :, :], J_local)
        outT[..., idj, :, :] = torch.matmul(outT[..., ipar, :, :], T_local)

    outJ = outT[..., :3, 3]
    Js = F.pad(Js, [0, 0, 0, 1])

    outT_0 = torch.matmul(outT, Js)
    # outT_0 = F.pad(outT_0, [3, 0, 0, 0, 0, 0, 0, 0])

    outT = outT - outT_0
    return outJ, outT


def analytical_normal(sigma, x, normalized=False):
    jacobian = torch.autograd.grad(
        outputs=sigma,
        inputs=x,
        grad_outputs=torch.ones_like(sigma),
        retain_graph=True)[0]

    normal = -jacobian
    if normalized:
        normal = F.normalize(normal, dim=-1)
    return normal

def ray_mesh_intersection(ray, mesh, gamma):
    """calculate the intersection between rays and mesh.

    Args:
        ray: tensor of shape `[..., 6]`, represents `[rays_o,rays_d]`.
        mesh: tensor of shape `[..., 3]`, the mesh vertices. Using Li Xiu's
        convention. Center points with axis vectors.
    Returns:
        A tensor of shape `[...,2]` represents the two intersection 
        depth. `[rays_o + t*rays_d]`. 
    """
    near = torch.full([ray.shape[0], mesh.shape[0]], 1e8)
    far = torch.full([ray.shape[0], mesh.shape[0]], -1.)
    rays_o, rays_d = ray[:, 0:3], ray[:, 3:6]
    z_0 = (mesh - rays_o[0, :]) @ rays_d.T
    ind = torch.norm(mesh - rays_o[0, :], dim=-1)[..., None] ** 2 - z_0 ** 2 < gamma ** 2
    z_delta = torch.sqrt(torch.clamp(gamma ** 2 - torch.norm(mesh - rays_o[0, :], dim=-1)[..., None] ** 2 + z_0 ** 2, 0))
    near = (z_0 - z_delta).masked_fill(~ind, 1e8)
    far = (z_0 + z_delta).masked_fill(~ind, -1)
    near = torch.min(near, dim=0)[0]
    far = torch.max(far, dim=0)[0]
    return torch.stack([near, far], dim=-1)
    

def ray_cube_intersection(ray, box, inf=1e8):
    """calculate the intersection between rays and a cube bounding-box.

    Args:
        ray: tensor of shape `[..., 6]`, represents `[rays_o,rays_d]`.
        box: tensor of shape `[6,3]`, the bounding-box. Using Li Xiu's
        convention. Center points with axis vectors.
        `[c_xy,c_yz,c_zx,v_x,v_y,v_z]^T`
        inf: the maximum depth. Default `1e8`        
    Returns:
        A tensor of shape `[...,2]` represents the two intersection 
        depth. `[rays_o + t*rays_d]`. 
    """
    near = torch.full([ray.shape[0], 6], inf)
    far = torch.full([ray.shape[0], 6], -1.)

    planes = []
    x = box[3, :]
    y = box[4, :]
    z = box[5, :]

    # Nx3
    c_xy = box[0, :]
    plane_xy = torch.cat([c_xy, x/2, y/2], dim=0)
    planes.append(plane_xy)
    plane_xy = torch.cat([c_xy+z, x/2, y/2], dim=0)
    planes.append(plane_xy)

    c_yz = box[1, :]
    plane_yz = torch.cat([c_yz, y/2, z/2], dim=0)
    planes.append(plane_yz)
    plane_yz = torch.cat([c_yz+x, y/2, z/2], dim=0)
    planes.append(plane_yz)

    c_zx = box[2, :]
    plane_zx = torch.cat([c_zx, z/2, x/2], dim=0)
    planes.append(plane_zx)
    plane_zx = torch.cat([c_zx+y, z/2, x/2], dim=0)
    planes.append(plane_zx)

    for idx, plane in enumerate(planes):
        tuv = ray_rectangle_intersection(ray, plane)
        id_valid = (torch.abs(tuv[..., 1]) <= 1) \
            * (torch.abs(tuv[..., 2]) <= 1)
        near[id_valid, idx] = tuv[id_valid, 0]
        far[id_valid, idx] = tuv[id_valid, 0]

    near = torch.min(near, dim=-1)[0]
    far = torch.max(far, dim=-1)[0]

    return torch.stack([near, far], dim=-1)


def line_plane_intersection(o, d, p0, p1, p2, eps=1e-12):
    r"""Calculate the intersection of line and plane

    Args:
        `o,d`: parameters of line `x=o+t*d`
        `p0,p1,p2`: parameters of the plane `x=p0+u*p1+v*p2`
    Returns:
        `[t,u,v]`: the intersections
    """
    denom = torch.sum(-d*torch.cross(p1, p2), dim=-1)
    t = torch.sum((o-p0)*torch.cross(p1, p2), dim=-1)
    u = torch.sum((o-p0)*torch.cross(p2.expand_as(d), -d), dim=-1)
    v = torch.sum((o-p0)*torch.cross(-d, p1.expand_as(d)), dim=-1)

    return torch.stack([t/(denom+eps), u/(denom+eps), v/(denom+eps)], dim=-1)


def ray_rectangle_intersection(ray, rects, eps=1e-8):
    o = ray[..., :3]
    d = ray[..., 3:]
    p0 = rects[..., :3]
    p1 = rects[..., 3:6]
    p2 = rects[..., 6:9]
    return line_plane_intersection(o, d, p0, p1, p2, eps)


def rodrigues(theta, eps=1e-8):
    """Rodrigues, Axis Angle to Quaternion.

    Args:
        theta: tensor of shape `[..., 3]`.
    Returns:
        quaternion: of shape `[...,4]` 
    """
    angle = torch.norm(theta + eps, p=2, dim=-1, keepdim=True)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=-1)
    return quat


def quat2mat(quat):
    """Quaternion to rotation matrix.    

    Args:
        quat: tensor of shape `[...,4]`.
    Returns:
        A tensor array of shape `[...,3,3]`
    """
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz, xy, xz, yz = w*x, w*y, w*z, x*y, x*z, y*z

    mat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=-1)

    return mat.view(quat.shape[:-1]+(3, 3))


def camera_rays(camK, W=None, H=None, c2w=None, graphics_coordinate=True,
                center=False):
    """shoot viewing rays from camera parameters.

    Args:
        camK: Tensor of shape `[3,3]`, the intrinsic matrix.
        W: Integer, if set None, then `W` is calculated as `2*cx`.
        H: Integer, if set None, then `H` is calculated as `2*cy`.
        c2w: Tensor of shape `[4,4]` or `[3,4]` camera view matrix. 
        If `None`, c2w is set as `[I,0]`
        graphics_coordinate: bool. Where or not use graphics coordinate 
        (pointing negative z into screen). Default: `True`.
        center: bool. Where or set 0.5 offset for pixels Default: `False`.

    Returns:
        rays_o: tensor of shape `[W,H,3]`  origins of the rays.
        rays_d: tensor of shape `[W,H,3]`  directions of the rays.
    """
    if c2w is None:
        c2w = torch.hstack((torch.eye(3), torch.zeros((3, 1))))
    if W is None:
        W = camK[0, 2]*2
    if H is None:
        H = camK[1, 2]*2
    W = int(W)
    H = int(H)

    invK = torch.inverse(camK)
    u, v = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    u = u.t()
    v = v.t()

    if center:
        u = u + 0.5
        v = v + 0.5
    dirs = torch.stack([u, v, torch.ones_like(u)], dim=-1)
    dirs = torch.matmul(dirs, invK.T)
    # use graphics coordinate. negtive z pointing into screen.
    if graphics_coordinate:
        dirs[..., 1] *= -1
        dirs[..., 2] *= -1

    rays_d = torch.matmul(dirs, c2w[:3, :3].T)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return torch.cat([rays_o, rays_d], dim=-1)


def rays_to_NDC(rays_o, rays_d, camK, near, W=None, H=None):
    r"""convert rays in word frame to NDC
    openGL projection matrix.
        [   
            [2fx/W, 0, 1-2cx/W, 0],\
            [0, -2fy/H, 1-2cy/H, 0],\
            [0, 0, -(f+n)/(f-n), -2fn/(f-n)],\
            [0, 0, -1, 0]
        ]
    We follow NeRF to set default far as inf. 
    and firstly move the zero plane to z=-near.
    """
    if W is None:
        W = int(2*camK[0, 2])
    if H is None:
        H = int(2*camK[1, 2])

    # Shift ray origins to near plane to avoid zero depth value.
    t_offset = -(near+rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t_offset[..., None]*rays_d

    # Projection
    o0 = -1./(W/(2.*camK[0, 0])) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*camK[1, 1])) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*camK[0, 0])) \
        * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*camK[1, 1])) \
        * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o_NDC = torch.stack([o0, o1, o2], -1)
    rays_d_NDC = torch.stack([d0, d1, d2], -1)

    return rays_o_NDC, rays_d_NDC, t_offset


def cast_rays(rays_o, rays_d, z_vals, r):
    """shoot viewing rays from camera parameters.

    Args:
        rays_o: tensor of shape `[...,3]`  origins of the rays.
        rays_d: tensor of shape `[...,3]`  directions of the rays.
        z_vals: tensor of shape [...,N] segments of the rays
        r: radius of ray cone. 1/f*2/\sqrt(12)
    Returns:
        mu: tensor of shape `[...,N,3]`  mean query positions
        cov_diag: tensor of shape `[...,N,3]`  corvirance of query 
        positions.
    """

    t0, t1 = z_vals[..., :-1], z_vals[..., 1:]
    c, d = (t0 + t1)/2, (t1 - t0)/2
    t_mean = c + (2*c*d**2) / (3*c**2 + d**2)
    t_var = (d**2)/3 - (4/15) * ((d**4 * (12*c**2 - d**2))
                                 / (3*c**2 + d**2)**2)
    r_var = r**2 * ((c**2)/4 + (5/12) * d**2 - (4/15)
                    * (d**4) / (3*c**2 + d**2))
    mu = rays_d[..., None, :] * t_mean[..., None]
    null_outer_diag = 1 - (rays_d**2) / \
        sum(rays_d**2, axis=-1, keepdims=True)
    cov_diag = (t_var[..., None] * (rays_d**2)[..., None, :]
                + r_var[..., None] * null_outer_diag[..., None, :])
    return mu + rays_o[..., None, :], cov_diag

def add_noise(pose: torch.Tensor, sigma: torch.Tensor):
    noise = torch.zeros_like(pose)
    if sigma.shape[0] > 1:
        for i in range(pose.shape[-1]):
            noise[:,i] = torch.randn(1) * sigma[i]
    else:
        noise = torch.randn(noise.size()) * sigma
    return pose + noise