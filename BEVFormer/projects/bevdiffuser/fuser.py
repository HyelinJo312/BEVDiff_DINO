import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import FUSERS
from mmcv.utils import Registry
# from mmdet3d.models.builder import FUSERS



@FUSERS.register_module()
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)

        # Output projection
        self.W_out = nn.Linear(self.d_v, d_model)

        # Optional: normalization
        self.norm = nn.GroupNorm(num_groups=1, num_channels=d_model)

    def forward(self, bev_feat, denoised_feat):
        """
        Args:
            bev_feat: [B, C, H, W] — original BEV feature (query)
            denoised_feat: [B, C, H, W] — denoised BEV feature from diffusion model (key & value)
        Returns:
            fused: [B, C, H, W] — refined BEV feature after cross-attention
        """
        B, C, H, W = bev_feat.shape

        # Flatten spatial dims
        bev_q = bev_feat.view(B, C, -1).permute(0, 2, 1)         # [B, HW, C]
        denoised_kv = denoised_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        # Linear projections
        Q = self.W_q(bev_q)             # [B, HW, d_k]
        K = self.W_k(denoised_kv)       # [B, HW, d_k]
        V = self.W_v(denoised_kv)       # [B, HW, d_v]

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, HW, HW]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)  # [B, HW, d_v]

        # Final linear projection and reshape
        fused = self.W_out(attn_output)              # [B, HW, d_model]
        fused = fused.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        # Add & Norm (residual connection)
        out = self.norm(fused + bev_feat)

        return out
    


@FUSERS.register_module()
class ConcatFusion_mlp(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=256, out_channels=128):
        """
        Args:
            in_channels (int): Channel count of each input feature
            hidden_dim (int): Hidden layer size in MLP
            out_channels (int, optional): Output channels (default = in_channels)
        """
        super().__init__()
        out_channels = out_channels or in_channels

        self.linear1 = nn.Linear(in_channels * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_channels)

        # BatchNorm2d will be applied after reshaping MLP output to (B, C, H, W)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: Tensor of shape (B, C, H, W)
            feat2: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, out_channels, H, W)
        """
        B, C, H, W = feat1.shape

        # Concatenate along channel dimension: (B, 2C, H, W)
        x = torch.cat([feat1, feat2], dim=1)  # (B, 2C, H, W)

        # Flatten to (B, H*W, 2C)
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, -1)

        # Apply MLP
        x = self.relu(self.linear1(x))  # (B, H*W, hidden_dim)
        x = self.linear2(x)             # (B, H*W, out_channels)

        # Reshape back to (B, out_channels, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  

        # Apply BatchNorm and activation
        x = self.bn(x)
        x = self.relu(x)

        return x
    


@FUSERS.register_module()
class ConcatFusion(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=256, out_channels=256, norm='group'):
        """
        1x1 Convolution for Fusion
        Args:
            in_channels (int): Channel count of each input feature
            hidden_dim (int): Hidden layer size in MLP
            out_channels (int, optional): Output channels (default = in_channels)
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_dim   = hidden_dim or in_channels

        self.conv1 = nn.Conv2d(in_channels*2, hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)

        if norm == 'group':
            self.norm = nn.GroupNorm(32, out_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()

        self.relu = nn.ReLU(inplace=False)

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)   # (B, 2C, H, W)
        x = self.relu(self.conv1(x))     # (B, hidden, H, W)
        x = self.conv2(x)                # (B, out, H, W)
        x = self.norm(x)
        x = self.relu(x)
        return x

