import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn
from .layers import DenseLayer
from .layers import PositionalEncoding

class Embedding(nn.Module):
    def __init__(self, smpl_model, body_shapes, body_poses, body_trans, latent_dim=16, embedding_res=10, output_dim=256, use_dir=False):
        super().__init__()
        # Hyperparameter
        self.smpl = smpl_model
        self.keypoint_dim = 7
        self.latent_dim = latent_dim
        self.use_dir = use_dir
        self.input_ch_embeded = (2 * embedding_res + 1) * 4

        # Smpl intialization
        self.body_betas = nn.Embedding.from_pretrained(body_shapes.clone(), freeze=True)
        self.body_poses = nn.Embedding.from_pretrained(body_poses.clone(), freeze=False)
        self.body_trans = nn.Embedding.from_pretrained(body_trans.clone(), freeze=True)

        # Positional Embedding
        self.embedder = PositionalEncoding(embedding_res)
        self.embedder_low = PositionalEncoding(4)
        
        # Learnable modules
        self.latent = nn.Parameter(torch.Tensor(smpl_model.th_v_template.shape[1], latent_dim))
        torch.nn.init.xavier_normal_(self.latent)
        self.aggregate = nn.Sequential(
            DenseLayer(self.keypoint_dim * (self.input_ch_embeded + self.latent_dim), output_dim, 'relu'),
            nn.ReLU(True),
            DenseLayer(output_dim, output_dim, 'relu'),
            nn.ReLU(True),
            DenseLayer(output_dim, output_dim, 'relu'),
        )

    def shape(self, idx):
        return self.body_betas.weight[idx].detach().clone()
    
    def pose(self, idx):
        return self.body_poses.weight[idx].detach().clone()

    def trans(self, idx):
        return self.body_trans.weight[idx].detach().clone()
        
    def forward(self, pts, theta, beta, trans=None, rot = None, keypoints = None, idx = None):
        rays, points, _ = pts.shape
        pts, view_dir = torch.split(pts, 3, dim=-1)
        if rot is not None:
            theta = torch.cat([rot, theta[:, 3:]], -1)
        if idx is None:
            keypoints, _, trans_kp = self.smpl(body_pose = theta, betas = beta, trans=trans)
            keypoints = keypoints[0]
            trans_kp = trans_kp[0]
        else:
            idx = torch.Tensor([idx]).long()
            keypoints, _, trans_kp = self.smpl(self.body_poses(idx), self.body_betas(idx), self.body_trans(idx))
            keypoints = keypoints[0]
            trans_kp = trans_kp[0]

        neighbors = self.smpl.neighbors
        rest_pose = self.smpl.th_v_template[0]

        # transformation w.r.t each vertex
        trans_kp = trans_kp.permute(2, 0, 1)
        trans_kp = torch.inverse(trans_kp)
        trans_kp = trans_kp.permute(1, 2, 0)
        trans_kp = trans_kp[:3, :3]

        # local features
        pts = pts.view(rays * points, -1)
        knn_index = knn(keypoints, pts, 1)
        knn_index = knn_index[1].view(rays * points)

        vertices_index = neighbors[knn_index]

        # distance
        vertices_dist = pts[..., None, :] - keypoints[vertices_index, :]

        # direction
        direction = (trans_kp[..., knn_index] * (pts - keypoints[knn_index]).T.unsqueeze(0)).sum(1).transpose(1, 0)
        direction = F.normalize(direction, dim=-1)

        # position
        vertices_feature = rest_pose[vertices_index, :].view(rays * points, -1)
        vertices_feature = torch.cat(
            [vertices_feature, torch.norm(vertices_dist, dim=-1)], dim=-1)

        # latent code
        latent_feature = self.latent[vertices_index]
        latent_feature = latent_feature.view(rays * points, -1)

        vertices_feature = self.embedder(vertices_feature)
        vertices_feature = self.aggregate(torch.cat([vertices_feature, latent_feature], -1))

        if self.use_dir:
            out_feature = torch.cat([vertices_feature, self.embedder_low(direction.view(rays * points, -1))], -1)
        else:
            out_feature = vertices_feature
        return out_feature.view(rays, points, -1)

class MNeRF(nn.Module):
    def __init__(self, layers=8, hidden_dim=256, input_dim=253):
        super(MNeRF, self).__init__()
        self.D = layers
        self.W = hidden_dim
        self.atten_ch = input_dim

        _nerf_layer = [DenseLayer(self.atten_ch,hidden_dim,'relu'),nn.ReLU(True)]
        for _ in range(layers-3):
            _nerf_layer += [DenseLayer(hidden_dim, hidden_dim, 'relu'), nn.ReLU(True)]
        self.nerf = nn.Sequential(*_nerf_layer)

        self.sigma_output = nn.Sequential(
            DenseLayer(hidden_dim, hidden_dim, activation='relu'),
            nn.ReLU(True),
            DenseLayer(hidden_dim, 1, activation='linear'))
        self.rgb_output = nn.Sequential(
            DenseLayer(hidden_dim, hidden_dim, activation='relu'),
            nn.ReLU(True),
            DenseLayer(hidden_dim, 3, activation='linear'))

    def alpha(self, inputs):
        batch_size = inputs.shape[:-1]
        inputs = inputs.view(-1, inputs.shape[-1])

        h = self.nerf(inputs)
        sigma = self.sigma_output(h)

        sigma = sigma.view(batch_size+(1,))
        outputs = {'sigma': sigma}
        return outputs

    def forward(self, inputs):
        batch_size = inputs.shape[:-1]
        inputs = inputs.view(-1, inputs.shape[-1])

        h = self.nerf(inputs)
        sigma = self.sigma_output(h)
        rgb = self.rgb_output(h)

        sigma = sigma.view(batch_size+(1,))
        rgb = rgb.view(batch_size+(3,))
        outputs = {'sigma': sigma, 'rgb': rgb}
        return outputs
