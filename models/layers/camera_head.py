import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res

class CameraHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        output_dim = dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_rot = nn.Linear(output_dim, 9)

    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat)

        # feat = self.avgpool(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())              ##########
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]
        out_t = self.fc_t(feat)  # [B,3]
        out_r = self.fc_rot(feat)  # [B,9]
        pose = self.convert_pose_to_4x4(BN, out_r, out_t, feat.device)

        return pose

    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))

        # CUDA SVD/determinant kernels are not implemented for bf16. Run the
        # orthogonalization path in float32 outside autocast, then restore the
        # caller's dtype for the rest of the model.
        original_dtype = m.dtype
        autocast_device = m.device.type if m.device.type in {"cuda", "cpu"} else "cuda"
        with torch.amp.autocast(device_type=autocast_device, enabled=False):
            m_float = m.float()
            m_transpose = torch.transpose(
                torch.nn.functional.normalize(m_float, p=2, dim=-1),
                dim0=-1,
                dim1=-2,
            )
            u, _, vh = torch.linalg.svd(m_transpose, full_matrices=False)
            v = vh.transpose(-2, -1)
            det = torch.linalg.det(torch.matmul(v, u.transpose(-2, -1)))
            # Flip the last singular vector when the decomposition reflects.
            r = torch.matmul(
                torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
                u.transpose(-2, -1),
            )
        return r.to(original_dtype)
