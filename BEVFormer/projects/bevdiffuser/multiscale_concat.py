import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConcat(nn.Module):
    def __init__(self, in_chs=(256,512,1024), out_dim=256, mid=256, use_concat=True):
        super().__init__()
        c0, c1, c2 = in_chs
        C = mid  
        
        self.layer0 = nn.Sequential(nn.Conv2d(c0, C, 1, bias=False), 
                                nn.GroupNorm(32, C), 
                                nn.SiLU())
        self.layer1 = nn.Sequential(nn.Conv2d(c1, C, 1, bias=False), 
                                nn.GroupNorm(32, C), 
                                nn.SiLU())
        self.layer2 = nn.Sequential(nn.Conv2d(c2, C, 1, bias=False), 
                                nn.GroupNorm(32, C), 
                                nn.SiLU())

        self.use_concat = use_concat
        in_mix = C*3 if use_concat else C
        self.mix = nn.Sequential(
            nn.Conv2d(in_mix, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.SiLU()
        )

    def forward(self, xs):
        x0, x1, x2 = xs
        B, _, H, W = x0.shape

        f0 = self.layer0(x0)                                   # [B,C,H,W]
        f1 = F.interpolate(self.layer1(x1), (H, W), mode='bilinear', align_corners=False)
        f2 = F.interpolate(self.layer2(x2), (H, W), mode='bilinear', align_corners=False)

        fused = torch.cat([f0, f1, f2], dim=1) if self.use_concat else (f0 + f1 + f2)/3.0
        return self.mix(fused)                              # [B,out_dim,H,W]
