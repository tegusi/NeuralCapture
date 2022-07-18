'''
This code is borrowed from smplpytorh(https://github.com/gulvarol/smplpytorch).
'''

import os

import chumpy as ch
from chumpy.ch import MatVecMult
import numpy as np
import cv2
import torch
import pickle
from torch.nn import Module
import torch.nn.functional as F

import torch
import scipy

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def th_get_axis_angle(vector):
    angle = torch.norm(vector, 2, 1)
    axes = vector / angle.unsqueeze(1)
    return axes, angle

def th_posemap_axisang(pose_vectors):
    '''
    Converts axis-angle to rotmat
    pose_vectors (Tensor (batch_size x 72)): pose parameters in axis-angle representation
    '''
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb):
        axis_ang = pose_vectors[:, joint_idx * 3:(joint_idx + 1) * 3]
        rot_mat = batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    rot_mats = torch.cat(rot_mats, 1)
    return rot_mats


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res


def subtract_flat_id(rot_mats):
    # Subtracts identity as a flattened tensor
    id_flat = torch.eye(
        3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(
            rot_mats.shape[0], 23)
    # id_flat.requires_grad = False
    results = rot_mats - id_flat
    return results


def make_list(tensor):
    # type: (List[int]) -> List[int]
    return tensor

class Rodrigues(ch.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel() for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel() for pp in p]).ravel()


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))

def ready_arguments(fname_or_dict):

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
        # dd = pickle.load(open(fname_or_dict, 'rb'))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas']) + dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd['pose']))

    return dd

def compute_norm(vertices, faces, point_buf):
    """
    Return:
        vertex_norm      -- torch.tensor, size (B, N, 3)
    Parameters:
        face_shape       -- torch.tensor, size (B, N, 3)
    """

    v1 = vertices[:, faces[:, 0]]
    v2 = vertices[:, faces[:, 1]]
    v3 = vertices[:, faces[:, 2]]
    e1 = v1 - v2
    e2 = v2 - v3
    face_norm = torch.cross(e1, e2, dim=-1)
    face_norm = torch.cross(e1, e2, dim=-1)
    face_norm = F.normalize(face_norm, dim=-1, p=2)
    face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3)], dim=1)
    
    vertex_norm = torch.sum(face_norm[:, point_buf], dim=2)
    vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
    # face_norm = face_norm / torch.norm(face_norm, dim=-1, keepdim=True)
    # face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 3)], dim=1)
    
    # vertex_norm = face_norm[:, point_buf, :]
    # vertex_norm = torch.sum(vertex_norm, dim=-2)
    # vertex_norm = torch.norm
    # vertex_norm = vertex_norm / torch.sum(vertex_norm, dim=-2, keepdim=True)
    # vertex_norm = torch.norm(vertex_norm, dim=-1)
    return vertex_norm

class SMPL(Module):
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints']

    def __init__(self,
                model_root,
                center_idx=None,
                gender='neutral'):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_root: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male'
        """
        super().__init__()

        self.center_idx = center_idx
        self.gender = gender

        if gender == 'neutral':
            self.model_path = os.path.join(model_root, 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')
        elif gender == 'female':
            self.model_path = os.path.join(model_root, 'basicmodel_f_lbs_10_207_0_v1.1.0.pkl')
        elif gender == 'male':
            self.model_path = os.path.join(model_root, 'basicmodel_m_lbs_10_207_0_v1.1.0.pkl')

        smpl_data = ready_arguments(self.model_path)
        self.smpl_data = smpl_data

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Kinematic chain params
        # self.graph = torch.zeros(self.th_v_template.shape[1], self.th_v_template.shape[1])
        # self.neighbors = torch.zeros(self.th_v_template.shape[1], self.th_v_template.shape[1])

        # Neighborhood Extraction
        self.neighbors = [{} for _ in range(self.th_v_template.shape[1])]
        self.point_buf = [[] for _ in range(self.th_v_template.shape[1])]
        for i in range(self.th_faces.shape[0]):
            self.point_buf[self.th_faces[i][0]].append(i)
            self.point_buf[self.th_faces[i][1]].append(i)
            self.point_buf[self.th_faces[i][2]].append(i)
            self.neighbors[self.th_faces[i][0]][int(self.th_faces[i][1])] = 1
            self.neighbors[self.th_faces[i][0]][int(self.th_faces[i][2])] = 1
            self.neighbors[self.th_faces[i][1]][int(self.th_faces[i][0])] = 1
            self.neighbors[self.th_faces[i][1]][int(self.th_faces[i][2])] = 1
            self.neighbors[self.th_faces[i][2]][int(self.th_faces[i][0])] = 1
            self.neighbors[self.th_faces[i][2]][int(self.th_faces[i][1])] = 1
        for i in range(len(self.point_buf)):
            while len(self.point_buf[i]) < 6:
                self.point_buf[i].append(self.point_buf[i][0])
            while len(self.point_buf[i]) > 6:
                self.point_buf[i].pop()
        self.point_buf = torch.tensor(self.point_buf)
        for i in range(len(self.neighbors)):
            self.neighbors[i] = list(self.neighbors[i])
            self.neighbors[i].append(i)
            while len(self.neighbors[i]) < 7:
                self.neighbors[i].append(self.neighbors[i][0])
            while len(self.neighbors[i]) > 7:
                self.neighbors[i].pop()
        self.neighbors = torch.tensor(self.neighbors)
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)  # 24

        # smpl_mesh_graph = np.load(os.path.join(model_root, 'mesh_downsampling.npz'), allow_pickle=True, encoding='latin1')
        # A = smpl_mesh_graph['A']
        # U = smpl_mesh_graph['U']
        # D = smpl_mesh_graph['D'] # shape: (2,)

        # downsampling
        # ptD = []
        # for i in range(len(D)):
        #     d = scipy.sparse.coo_matrix(D[i])
        #     i = torch.Tensor(np.array([d.row, d.col])).long()
        #     v = torch.Tensor(d.data)
        #     ptD.append(torch.sparse_coo_tensor(i, v, d.shape))
        
        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890]
        # ptD[1].to_dense() - Size: [431. 1723]
        # Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense()) # 6890 -> 431
        # Dmap.requires_grad = True
        # self.register_buffer('Dmap', Dmap)

    def forward(self,
                body_pose,
                betas=torch.zeros(1),
                trans=torch.zeros(1),
                h2w = None):
        """
        Args:
        body_pose (Tensor (batch_size x 72)): pose parameters in axis-angle representation
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters
        th_trans (Tensor (batch_size x 3)): if provided, applies trans to joints and vertices
        """

        batch_size = body_pose.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(body_pose)
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat)

        # Below does: v_shaped = v_template + shapedirs * betas
        # If shape parameters are not provided
        if betas is None or bool(torch.norm(betas) == 0):
            th_v_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)
        else:
            th_v_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, betas.transpose(1, 0)).permute(2, 0, 1)
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)

        # Below does: v_posed = v_shaped + posedirs * pose_map
        th_v_posed = th_v_shaped + torch.matmul(
            self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        # Final T pose with transformation done!

        # Global rigid transformation
        th_results = []

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val *
                                   9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results

        th_results2 = torch.zeros((batch_size, 4, 4, self.num_joints),
                                  dtype=root_j.dtype,
                                  device=root_j.device)

        for i in range(self.num_joints):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]

        # If translation is not provided
        if trans is None or bool(torch.norm(trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + trans.unsqueeze(1)
            th_verts = th_verts + trans.unsqueeze(1)
        if h2w is not None:
            h2w_R = h2w[..., :3, :3]
            h2w_T = h2w[..., :3, 3:]
            th_verts = torch.bmm(h2w_R, th_verts.transpose(-1, -2)) + h2w_T
            th_verts = th_verts.transpose(-1, -2)
        # norm = compute_norm(th_verts, self.th_faces, self.point_buf)
        # norm_t = compute_norm(th_v_posed, self.th_faces, self.point_buf)
        # Vertices and joints in meters
        return th_verts, th_jtr, th_T