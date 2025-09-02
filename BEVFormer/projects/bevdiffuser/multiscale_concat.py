import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConcat(nn.Module):
    """
    - x0: target (예: out_list[0], [B,256,50,50]) → 아무 처리 없이 사용
    - x1: (예: [B,512,25,25]) → Conv+GN+ReLU → 업샘플(50x50)
    - x2: (예: [B,1024,12,12]) → Conv+GN+ReLU → 업샘플(50x50)
    - concat 후 1x1 Conv(+GN+ReLU)로 out_dim으로 압축
    """
    def __init__(self,
                 in_chs=(256, 512, 1024),   # (x0,x1,x2 채널)
                 out_dim=256,
                 pick_idxs=(0, 1, 2),      
                 target_idx=0):            
        super().__init__()
        self.pick_idxs = pick_idxs
        self.target_idx = target_idx
        c0, c1, c2 = in_chs

        self.layer1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, c1),
            nn.ReLU(),   # nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, c2),
            nn.ReLU(),   # nn.ReLU(inplace=True)
        )

        # concat 후 채널 압축
        self.out_layer = nn.Sequential(
            nn.Conv2d(c0 + c1 + c2, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),   # nn.ReLU(inplace=True)
        )

    def forward(self, multi_feats):
   
        x0 = multi_feats[self.pick_idxs[0]]                        # [B, c0, H, W]
        B, _, H, W = x0.shape

        x1 = multi_feats[self.pick_idxs[1]]                        # [B, c1, h1, w1]  h1, w1 = H/2, W/2
        x2 = multi_feats[self.pick_idxs[2]]                        # [B, c2, h2, w2]  h2, w2 = H/4, W/4
 
        # x1 = self.layer1(x1)
        # x2 = self.layer2(x2)

        if x1.shape[-2:] != (H, W):
            x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=False)
        if x2.shape[-2:] != (H, W):
            x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=False)

        # concat → 1x1 for chanel dimension
        fused = torch.cat([x0, x1, x2], dim=1)
        fused = self.out_layer(fused)
        return fused
