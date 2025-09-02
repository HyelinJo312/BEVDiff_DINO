import torch.nn as nn
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from timm.models.layers import trunc_normal_
from omegaconf import OmegaConf
import torch.nn.functional as F
from einops import repeat
import os
import torchvision.transforms as T
from collections import OrderedDict
# import lightning as L
from typing import Optional
import math
# from utils import pad, unpad, silog
# from optimizer import get_optimizer
# from metrics import compute_metrics
# from utils import eigen_crop, garg_crop, custom_crop, no_crop

NUM_DECONV = 3
NUM_FILTERS = [32, 32, 32]
DECONV_KERNELS = [2, 2, 2]
VIT_MODEL = 'google/vit-base-patch16-224'



def pad_to_make_square(x):
    y = 255*((x+1)/2)
    y = torch.permute(y, (0,2,3,1))
    bs, _, h, w = x.shape
    if w>h:
        patch = torch.zeros(bs, w-h, w, 3).to(x.device)
        y = torch.cat([y, patch], axis=1)
    else:
        patch = torch.zeros(bs, h, h-w, 3).to(x.device)
        y = torch.cat([y, patch], axis=2)
    return y.to(torch.int)


class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts



class CIDE(nn.Module):
    def __init__(self, args, emb_dim, train_from_scratch):
        super().__init__()
        self.args = args
        self.vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL, resume_download=True)
        if train_from_scratch:
            vit_config = ViTConfig(num_labels=1000)
            self.vit_model = ViTForImageClassification(vit_config)
        else:
            self.vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL, resume_download=True)
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(1000, 400),
            nn.GELU(),
            nn.Linear(400, args.no_of_classes)
        )
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
        
        self.embeddings = nn.Parameter(torch.randn(self.args.no_of_classes, self.dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)
    
    def forward(self, x):
        y = pad_to_make_square(x)
        # use torch.no_grad() to prevent gradient flow through the ViT since it is kept frozen
        with torch.no_grad():
            inputs = self.vit_processor(images=y, return_tensors="pt").to(x.device)
            vit_outputs = self.vit_model(**inputs)
            vit_logits = vit_outputs.logits
            
        class_probs = self.fc(vit_logits)
        class_probs = self.m(class_probs)
        
        class_embeddings = class_probs @ self.embeddings
        conditioning_scene_embedding = self.embedding_adapter(class_embeddings, self.gamma) 
        
        return conditioning_scene_embedding
    

    
class GetDINOv2Cond(nn.Module):
    """
    DINOv2 condition encoder (HF-only, Tensor-only).

    Inputs:
      - images: torch.Tensor of shape (B, C, H, W) or (B, V, C, H, W), RGB
                values can be in [0,1] or [0,255]

    Returns (preserves view dim if provided):
      - 'cls'   : (B, C_dino) or (B, V, C_dino)          # global embedding (pooler_output)
      - 'cond'  : (B, features) or (B, V, features)      # projected conditioning
      - 'tokens': (B, 1+N, C_dino) or (B, V, 1+N, C_dino)  (optional; last_hidden_state)
    """
    def __init__(
        self,
        encoder: str = 'vitb',     # ['vits', 'vitb', 'vitl'] → small/base/large
        features: int = 256,       # output dim for BEVDiffuser conditioning
        device: str = 'cuda',
        freeze: bool = True,
        patch: int = 14,
        input_size: int = 518,     # ViT/14 권장 정사각 입력
        pretrained: bool = True,
        symmetric_pad: bool = True,
    ):
        super().__init__()
        assert encoder in ['vits', 'vitb', 'vitl']
        self.device = device
        self.input_min = input_size
        self.patch = patch
        self.symmetric_pad = symmetric_pad

        from transformers import AutoImageProcessor, Dinov2Model

        self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size  # 384/768/1024

        # projection to BEVDiffuser conditioning dim
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, features),
            nn.GELU(),
            nn.Linear(features, features),
        ).to(device)

        # ImageNet mean/std buffers (for in-graph normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer('imgnet_mean', mean, persistent=False)
        self.register_buffer('imgnet_std', std, persistent=False)


    def image_preprocess(self, x):
        """
        x: (N, 3, H, W) float tensor in [0,1] or [0,255]
        -> (x_norm: (N, 3, H2, W2) on self.device, ImageNet normalized,
            Hp, Wp: patch grid size = (H2//patch, W2//patch))
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected (N,3,H,W), got {x.shape}"
        x = x.to(dtype=torch.float32)
        if x.max() > 1.5:      # likely [0,255]
            x = x / 255.0
        x = x.clamp_(0.0, 1.0)

        N, C, H, W = x.shape
        # 종횡비 유지: min(H1, W1) >= input_min
        scale = max(self.input_min / H, self.input_min / W)
        H1, W1 = int(round(H * scale)), int(round(W * scale))

        # 리사이즈
        x = F.interpolate(x, size=(H1, W1),
                          mode='bicubic', align_corners=False)

        # 14 배수 정렬 (ceil) → 패딩
        H2 = (H1 + self.patch - 1) // self.patch * self.patch
        W2 = (W1 + self.patch - 1) // self.patch * self.patch
        pad_h, pad_w = H2 - H1, W2 - W1

        if self.symmetric_pad:
            top = pad_h // 2; bottom = pad_h - top
            left = pad_w // 2; right = pad_w - left
        else:
            top = 0; bottom = pad_h; left = 0; right = pad_w

        x = F.pad(x, (left, right, top, bottom), mode='replicate')

        x = (x - self.imgnet_mean) / self.imgnet_std
        x = x.to(self.device, non_blocking=True)

        Hp, Wp = H2 // self.patch, W2 // self.patch
        return x, Hp, Wp
    

    # def _preprocess(self, x: torch.Tensor) -> torch.Tensor: 
    #     """ x: (Bflat, 3, H, W) float tensor in [0,1] or [0,255] 
    #     -> (Bflat, 3, S, S) normalized to ImageNet stats 
    #     """ 

    #     x = x.to(self.device, dtype=torch.float32) 
    #     if x.max() > 1.5: 
    #         # if likely in [0,255] 
    #         x = x / 255.0 
    #     if (x.shape[-2] != self.input_min) or (x.shape[-1] != self.input_min): 
    #         x = F.interpolate(x, size=(self.input_min, self.input_min), mode='bicubic', align_corners=False) 

    #     x = (x - self.imgnet_mean) / self.imgnet_std 
    #     return x
    

    @torch.no_grad()
    def forward(self, images, img_metas, n_layers=4):
        """
        images: (B, 6, C, H, W) or (6, C, H, W) when bs=1
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")

        if images.ndim == 5:
            B, V, C, H, W = images.shape
            x = images.reshape(B * V, C, H, W).contiguous()
        elif images.ndim == 4:
            V, C, H, W = images.shape
            B  = 1
            x = images.reshape(B * V, C, H, W).contiguous()
        else:
            raise ValueError(f"Uneimgspected tensor shape: {images.shape}")
        
        # x = self._preprocess(x)
        x, Hp, Wp = self.image_preprocess(x)

        # Dinov2 forward: pass as pixel_values
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hs_selected = hidden_states[-n_layers:] if n_layers > 0 else [hidden_states[-1]]

        feats_out = []
        for h in hs_selected:
            cls_tok = h[:, 0]          # (B*V, C_dino)
            tok    = h[:, 1:]          # (B*V, Hp*Wp, C_dino)
            # tok = h[:, 1:, :] 
            cls_tok = cls_tok.view(B, V, self.hidden_dim)                  # (B,V,C)
            # cond = self.proj(cls_tok)   # 예: 글로벌 conditioning이 필요할 때
            tok_seq = tok.view(B, V, Hp * Wp, self.hidden_dim)             # (B,V,N,C)
            feats_out.append((tok_seq, cls_tok))

        last_tok, last_cls = feats_out[-1][0], feats_out[-1][1]

        return {
            'features': feats_out,          # list[(B,V,N,C),(B,V,C)]
            'patch_hw': (Hp, Wp),
            'last_cls': last_cls,           # (B, V, C)
            'last_tokens': last_tok,        # (B, V, N, C)
            'img_metas': img_metas
        }






# class GetDINOv2Cond(nn.Module):
#     """
#     DINOv2 condition encoder (HF-only, Tensor-only).

#     Inputs:
#       - images: torch.Tensor of shape (B, C, H, W) or (B, V, C, H, W), RGB
#                 values can be in [0,1] or [0,255]

#     Returns (preserves view dim if provided):
#       - 'cls'   : (B, C_dino) or (B, V, C_dino)          # global embedding (pooler_output)
#       - 'cond'  : (B, features) or (B, V, features)      # projected conditioning
#       - 'tokens': (B, 1+N, C_dino) or (B, V, 1+N, C_dino)  (optional; last_hidden_state)
#     """
#     def __init__(
#         self,
#         encoder='vitb',     # ['vits', 'vitb', 'vitl'] → small/base/large
#         features=256,       # output dim for BEVDiffuser conditioning
#         device='cuda',
#         freeze=True,
#         input_size=518,     # ViT/14 권장 정사각 입력
#         patch=14,
#         symmetric_pad=True,
#     ):
#         super().__init__()
#         assert encoder in ['vits', 'vitb', 'vitl']
#         self.device = device
#         self.input_size = input_size
#         self.patch = patch
#         self.symmetric_pad = symmetric_pad

#         self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#         for p in self.model.parameters():
#             p.requires_grad = False
#         self.model.eval()

#     @staticmethod
#     def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
#         mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
#         std  = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
#         return (x - mean) / std

#     def _preprocess(self, x):
#         """
#         x: (N, 3, H, W) float [0,1] or [0,255] on any device
#         -> (N, 3, H2, W2) on self.device, ImageNet normalized
#            plus (Hp, Wp) = (H2//patch, W2//patch)
#         """
#         x = x.to(torch.float32, non_blocking=True)
#         if x.max() > 1.5:
#             x = x / 255.0
#         x = x.clamp_(0.0, 1.0)

#         N, C, H, W = x.shape
#         # aspect-ratio preserving scale so that min side >= input_min
#         scale = max(self.input_min / H, self.input_min / W)
#         H1, W1 = int(round(H * scale)), int(round(W * scale))

#         x = F.interpolate(x, size=(H1, W1), mode='bicubic',
#                           align_corners=False, antialias=True)

#         # ceil to patch multiple by padding
#         H2 = (H1 + self.patch - 1) // self.patch * self.patch
#         W2 = (W1 + self.patch - 1) // self.patch * self.patch
#         pad_h, pad_w = H2 - H1, W2 - W1

#         if self.symmetric_pad:
#             top = pad_h // 2
#             bottom = pad_h - top
#             left = pad_w // 2
#             right = pad_w - left
#         else:
#             top = 0; left = 0
#             bottom = pad_h; right = pad_w

#         x = F.pad(x, (left, right, top, bottom), mode='replicate')

#         x = self._imagenet_norm(x).to(self.device, non_blocking=True)
#         Hp, Wp = H2 // self.patch, W2 // self.patch
#         return x, Hp, Wp
    
    
#     @torch.no_grad()
#     def forward(self, x, img_metas, n_layers):
#         """
#         images: (B, 6, C, H, W) or (6, C, H, W) when bs=1
#         """
#         if not isinstance(x, torch.Tensor):
#             raise TypeError("x must be a torch.Tensor")

#         if x.ndim == 5:
#             B, V, C, H, W = x.shape
#             x = x.reshape(B * V, C, H, W).contiguous()
#         elif x.ndim == 4:
#             V, C, H, W = x.shape
#             B  = 1
#             x = x.reshape(B * V, C, H, W).contiguous()
#         else:
#             raise ValueError(f"Uneimgspected tensor shape: {x.shape}")

#         pixel_values, Hp, Wp = self._preprocess(x)

#         inter = self.model.get_intermediate_layers(pixel_values, n=n_layers, reshape=False, return_class_token=True)

#         feats = []
#         for x_l in inter:
#             cls = x_l[:, 0]          # (B, C)
#             patch = x_l[:, 1:]       # (B, Hp*Wp, C)
#             patch = patch.view(pixel_values.size(0), Hp, Wp, -1).permute(0,3,1,2)  # (B,C,Hp,Wp)
#             feats.append({"cls": cls, "patch": patch})

#         return {
#             'features': feats,          # list[(B,V,N,C),(B,V,C)]
#             'patch_hw': (Hp, Wp),
#             # 'last_cls': last_cls,           # (B, V, C)
#             # 'last_tokens': last_tok,        # (B, V, N, C)
#             'img_metas': img_metas
#         }

        