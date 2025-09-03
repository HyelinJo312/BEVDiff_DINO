# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVFormer
#   (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) 2022 BEVFormer authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import copy
import torch
import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS, FUSERS
from mmcv.utils import ConfigDict
from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head import BEVFormerHead
from projects.bevdiffuser.model_utils import build_unet, instantiate_from_config
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import Registry, build_from_cfg
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler
from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler
from projects.bevdiffuser.layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel
from projects.bevdiffuser.multiscale_concat import MultiScaleConcat
from projects.bevdiffuser.fuser import CrossAttentionFusion, ConcatFusion
# FUSERS = Registry('fuser')
# FUSERS.register_module(CrossAttentionFusion)
# FUSERS.register_module(ConcatFusion)



@HEADS.register_module()
class BEVDiffuserHead(BEVFormerHead):
    """Head of BEVDiffuser with Fusion

          Encoder -> BEV (B, HW, C) -> (B,C,H,W)
          -> add noise (train) -> UNet -> denoised (B,C,H,W)
          -> Fuser(original, denoised) -> fused (B,C,H,W) -> (B,HW,C)
          -> Decoder (given_bev=fused) -> Detection losses
    """

    def __init__(self,
                 *args,
                 unet=None,
                 fuser=None,
                 noise_scheduler=None,
                #  infer_scheduler=None,
                 denoise_loss_weight=1.0,
                 return_multiscale=False,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        if unet is None:
            raise ValueError("`unet` config is required for BEVDiffuserHead")
        # self.unet = build_unet(ConfigDict(unet))
        self.unet = instantiate_from_config(ConfigDict(unet))
        pretrained = unet.get('pretrained', None)
        if pretrained:
            state = torch.load(pretrained, map_location='cpu')
            self.unet.load_state_dict(state, strict=False)

        self.fuser = build_from_cfg(fuser, FUSERS) if isinstance(fuser, dict) else fuser

        self.denoise_loss_weight = float(denoise_loss_weight)
        self.return_multiscale = return_multiscale
        self.multi_scale_concat = MultiScaleConcat(in_chs=(256, 512, 1024), out_dim=256, pick_idxs=(0, 1, 2), target_idx=0)
        
        # Load scheduler, tokenizer and models.
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            noise_scheduler.from_pretrained, subfolder=noise_scheduler.subfolder
        )
        self.infer_scheduler = DDIMGuidedScheduler.from_pretrained(
            noise_scheduler.from_pretrained, subfolder=noise_scheduler.subfolder
        )
        
        if noise_scheduler.prediction_type == 'sample':
            self.noise_scheduler.register_to_config(prediction_type='sample')
            self.infer_scheduler.register_to_config(prediction_type='sample')

        # self.noise_scheduler = self._build_noise_scheduler(noise_scheduler)
        # self.infer_scheduler = self._build_noise_scheduler(infer_scheduler)

        # self._disable_inplace(self)

    def _build_noise_scheduler(self, cfg):
        if cfg is None:
            return None
        t = cfg.get('type', '')
        if t == 'DDPMScheduler':
            from diffusers import DDPMScheduler
            sch = DDPMScheduler.from_pretrained(
                cfg['from_pretrained'],
                subfolder=cfg.get('subfolder', 'scheduler'))
            if 'prediction_type' in cfg and cfg['prediction_type'] is not None:
                sch.register_to_config(prediction_type=cfg['prediction_type'])
            return sch
        elif t == 'DDIMGuidedScheduler':
            from projects.bevdiffuser.scheduler_utils import DDIMGuidedScheduler
            sch = DDIMGuidedScheduler.from_pretrained(
                cfg['from_pretrained'],
                subfolder=cfg.get('subfolder', 'scheduler'))
            if 'prediction_type' in cfg and cfg['prediction_type'] is not None:
                sch.register_to_config(prediction_type=cfg['prediction_type'])
            return sch
        else:
            raise ValueError(f'Unknown noise scheduler type: {t}')


    @staticmethod
    def _disable_inplace(module: nn.Module):
        # ReLU(inplace=True) 같은 것을 전부 inplace=False로 바꿔 autograd 안전성 확보
        for m in module.modules():
            if isinstance(m, nn.ReLU) and getattr(m, 'inplace', False):
                m.inplace = False
        

    def compute_denoise(self, latents, **cond):
        assert self.noise_scheduler is not None, \
            'noise_scheduler is required for training the diffuser'
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        bsz = latents.shape[0]
        max_timestep = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, max_timestep, (bsz,), device=latents.device).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        ptype = getattr(self.noise_scheduler.config, 'prediction_type', 'epsilon')
        if ptype == 'epsilon':
            target = noise
        elif ptype == 'sample':
            target = latents
        elif ptype == 'v_prediction':
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f'Unknown prediction_type: {ptype}')

        # Multi-scale features
        if self.return_multiscale:  # UNet input: noisy_latents=(B, C, H, W)
            denoise_results = self.unet(noisy_latents, timesteps, **cond) # layout diffusion: self.unet(noisy_latents, timesteps, **cond)[0] /  DINO diffusion: self.unet(noisy_latents, timesteps, **cond)
            multi_feats = self.multi_scale_concat(denoise_results)
            pred = denoise_results[0]
        else:
            # Predict the noise residual and compute loss
            pred = self.unet(noisy_latents, timesteps, **cond)

        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction='mean')
        # return pred, denoise_loss
        return multi_feats, denoise_loss



    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, dino_feats=None, **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return bev_embed

        B, HW, C = bev_embed.shape
        H, W = self.bev_h, self.bev_w

        # (B, HW, C) -> (B, C, H, W)
        bev_embed = bev_embed.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        latents = bev_embed.clone().detach()

        cond = dino_feats

        denoise_loss = None

        # Training: add noise -> UNet -> denoised
        # if return_loss:
        if self.training:
            multi_feats, denoise_loss = self.compute_denoise(latents, **cond)
            denoised_bev = multi_feats    # B, C, H, W
        # Evaluation: DDIM sampling
        else:
            with torch.no_grad():
                denoised_bev = self.denoise_eval(latents, cond, img_metas, **kwargs)

        # Fuser: (original, denoised) -> fused (B,C,H,W)
        fused_bev = self.fuser(bev_embed, denoised_bev) if self.fuser is not None else denoised_bev
        fused_bev = fused_bev.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # (B, HW, C)

        outputs = self.transformer(
            mlvl_feats,
            bev_queries, 
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
            given_bev=fused_bev  # (B, HW, C)
        )

        bev_embed_out, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed_out,  # HW, B, C
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        if denoise_loss is not None:
            outs['denoise_loss'] = denoise_loss

        return outs


    def denoise_eval(self, latents, cond, img_metas, **kwargs):
        """
        Evaluation-time denoising (DDIM + CFG + optional task-guidance).
        Args:
            latents: (B, C, H, W) encoder BEV feature (no grad path)
            cond:    conditioning dict for UNet
            img_metas, **kwargs: passed down for optional task guidance
        Returns:
            denoised_bev: (B, C, H, W)
        """
        
        if (self.unet is None) or (self.infer_scheduler is None):
            return latents

        diff_cfg = getattr(self, 'test_cfg', {}).get('diffusion', {})
        noise_timesteps     = int(diff_cfg.get('noise_timesteps', 0) or 0)
        denoise_timesteps   = int(diff_cfg.get('denoise_timesteps', 0) or 0)
        num_inference_steps = int(diff_cfg.get('num_inference_steps', 0) or 0)
        ddim_sampling_eta = float(diff_cfg.get('ddim_sampling_eta', 1.0) or 0.0)
        guidance_scale      = float(diff_cfg.get('guidance_scale', 2.0))
        use_task_guidance   = bool(diff_cfg.get('use_task_guidance', False))

        def get_dino_uncond(dino_out):
            uncond = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in cond.items()}
            last_cls_u = torch.zeros_like(dino_out['last_cls'])  # (B,V,C_in)
            last_tokens_u = torch.zeros_like(dino_out['last_tokens'])  # (B,V,N,C_in)
            uncond['last_cls'] = last_cls_u
            uncond['last_tokens'] = last_tokens_u
            return uncond

        if noise_timesteps > 0:
            if noise_timesteps > 1000:
                latents = torch.randn_like(latents)
                if hasattr(self.infer_scheduler, 'init_noise_sigma'):
                    latents = latents * self.infer_scheduler.init_noise_sigma
            else:
                noise = torch.randn_like(latents)
                noise_timesteps = torch.tensor(noise_timesteps, device=latents.device, dtype=torch.long)
                latents = self.infer_scheduler.add_noise(latents, noise, noise_timesteps)

        # DDIM sampling
        if denoise_timesteps > 0 and num_inference_steps > 0:
            uncond = get_dino_uncond(cond) 

            self.infer_scheduler.config.num_train_timesteps = int(denoise_timesteps)
            self.infer_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

            for t in self.infer_scheduler.timesteps:
                t_batch = torch.full((latents.shape[0],), int(t), device=latents.device, dtype=torch.long)
                noise_pred_uncond = self.unet(latents, t_batch, **uncond)[0]
                noise_pred_cond   = self.unet(latents, t_batch, **cond)[0]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                classifier_gradient = None
                if use_task_guidance:
                    classifier_gradient = self._classifier_guidance_grad(latents, img_metas=img_metas, **kwargs)
                latents = self.infer_scheduler.step(noise_pred, t, latents, eta=ddim_sampling_eta, return_dict=False, classifier_gradient=classifier_gradient)[0]

        return latents



    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        if pos_inds.numel() != 0:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        if pos_inds.numel() != 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        
        if normalized_bbox_targets[isnotnan, :10].numel() == 0:
            loss_bbox = torch.tensor(0., dtype=torch.float32).to(bbox_preds.device)
        else:
            loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
                avg_factor=num_total_pos)
            
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        # denoise loss
        if 'denoise_loss' in preds_dicts and preds_dicts['denoise_loss'] is not None:
            loss_dict['loss_diff'] = preds_dicts['denoise_loss'] * self.denoise_loss_weight

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list



class BEVFormerHead_GroupDETR(BEVFormerHead):
    def __init__(self,
                 *args,
                 group_detr=1,
                 **kwargs):
        self.group_detr = group_detr
        assert 'num_query' in kwargs
        kwargs['num_query'] = group_detr * kwargs['num_query']
        super().__init__(*args, **kwargs)

    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:  # NOTE: Only difference to bevformer head
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        assert enc_cls_scores is None and enc_bbox_preds is None 

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        loss_dict = dict()
        loss_dict['loss_cls'] = 0
        loss_dict['loss_bbox'] = 0
        for num_dec_layer in range(all_cls_scores.shape[0] - 1):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = 0
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = 0
        num_query_per_group = self.num_query // self.group_detr
        for group_index in range(self.group_detr):
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index+1) * num_query_per_group
            group_cls_scores =  all_cls_scores[:, :,group_query_start:group_query_end, :]
            group_bbox_preds = all_bbox_preds[:, :,group_query_start:group_query_end, :]
            losses_cls, losses_bbox = multi_apply(
                self.loss_single, group_cls_scores, group_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list)
            loss_dict['loss_cls'] += losses_cls[-1] / self.group_detr
            loss_dict['loss_bbox'] += losses_bbox[-1] / self.group_detr
            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls'] += loss_cls_i / self.group_detr
                loss_dict[f'd{num_dec_layer}.loss_bbox'] += loss_bbox_i / self.group_detr
                num_dec_layer += 1
        return loss_dict