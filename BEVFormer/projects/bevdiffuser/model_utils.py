# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVFormer
#   (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) 2022 BEVFormer authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import torch
import importlib
from mmcv import Config
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset



def get_bev_model(args):
    cfg = Config.fromfile(args.bev_config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'): 
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.bev_config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # init_dist(args.launcher, **cfg.dist_params)
        
    cfg.model.pretrained = None
    model = build_model(cfg.model, 
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.bev_checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    # model.eval()
    return model


def get_bev_model_scratch(args):
    """
    Build a BEV model from scratch (except image backbone pretrained) for end-to-end training.
    Does NOT load any pretrained weights or checkpoints.
    """
    cfg = Config.fromfile(args.bev_config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'): 
        if cfg.plugin:
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.bev_config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True

    # Ensure model is initialized from scratch
    # cfg.model.pretrained = None
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )

    # Apply fp16 if configured
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    model.init_weights()

    dataset_default_args = {
        'pc_range': cfg.point_cloud_range,
        'use_3d_bbox': cfg.use_3d_bbox,
        'num_classes': cfg.num_classes,
        'num_bboxes': cfg.num_bboxes,
    }
    datasets = [build_dataset(cfg.data.train,
                              default_args=dataset_default_args)]
    model.CLASSES = datasets[0].CLASSES

    # # Wrap for data parallel or distributed
    # if not distributed:
    #     model = MMDataParallel(model, device_ids=[0])
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False,
    #         find_unused_parameters=True
    #     )
    return model

def build_unet(cfg):
    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)

        return getattr(importlib.import_module(module, package=None), cls)
    
    layout_encoder = get_obj_from_str(cfg.parameters.layout_encoder.type)(
        **cfg.parameters.layout_encoder.parameters
    )

    model_kwargs = dict(**cfg.parameters)
    model_kwargs.pop('layout_encoder')
    return get_obj_from_str(cfg.type)(
        layout_encoder=layout_encoder,
        **model_kwargs,
    )


def instantiate_from_config(cfg):
    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
            
        return getattr(importlib.import_module(module, package=None), cls)
    
    return get_obj_from_str(cfg.type)(**cfg.get("parameters", dict()))


# Load scratch model
def get_bev_model_v2(args):
    import os
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    from mmcv.runner import wrap_fp16_model, load_checkpoint
    from mmdet.models import build_detector as build_model

    cfg = Config.fromfile(args.bev_config)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if getattr(cfg, 'plugin', False):
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir).split('/')
        else:
            _module_dir = os.path.dirname(args.bev_config).split('/')
        _module_path = _module_dir[0]
        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        importlib.import_module(_module_path)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    distributed = (args.launcher != 'none')

    # 목표: 이미지 백본의 pretrained(dict(img=...))는 유지,
    # 그 외는 scratch 학습 시 랜덤 초기화(or 서브모듈의 pretrained None) 되도록.
    def _keep_img_pretrained_on_model(cfg_model):
        if 'pretrained' in cfg_model:
            if isinstance(cfg_model['pretrained'], dict) and ('img' in cfg_model['pretrained']):
                # ex) pretrained=dict(img='torchvision://resnet50') 유지
                pass
            else:
                cfg_model['pretrained'] = None

        for k in ['img_backbone', 'backbone', 'img_neck', 'neck', 'pts_backbone']:
            if k in cfg_model and isinstance(cfg_model[k], dict):
                if 'pretrained' in cfg_model[k]:
                    # 이미지 백본 계열은 유지(값이 비어있지 않다면)
                    if k in ['img_backbone', 'backbone', 'img_neck'] and cfg_model[k]['pretrained']:
                        pass  # keep
                    else:
                        # 그 외는 scratch 시 None 권장
                        cfg_model[k]['pretrained'] = None

    # scratch 조건: train_bev_from_scratch=True 이거나 bev_checkpoint가 비어있음
    scratch_mode = (
        getattr(args, 'train_bev_from_scratch', False) or
        (getattr(args, 'bev_checkpoint', None) in [None, ''] or
         not (os.path.isfile(getattr(args, 'bev_checkpoint', '')) or os.path.isdir(getattr(args, 'bev_checkpoint', ''))))
    )

    if scratch_mode:  # 이미지 백본의 pretrained(dict(img=...))만 보존
        if hasattr(cfg, 'model') and isinstance(cfg.model, dict):
            _keep_img_pretrained_on_model(cfg.model)
    else:
        # 체크포인트 로드 경로면 굳이 손대지 않아도 됨 (ckpt가 덮어씀)
        pass

    model = build_model(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    if scratch_mode:
        # 체크포인트를 로드하지 않음. init_weights를 호출하여
        # - 랜덤 초기화
        # - (config에 남겨둔) 이미지 백본 pretrained는 백본 init 시 자동 반영
        if hasattr(model, 'init_weights'):
            model.init_weights()
        print("[get_bev_model] Scratch training: no BEV checkpoint is loaded. "
              "Image backbone pretrained from config is preserved (if specified).")
    else:
        checkpoint = load_checkpoint(model, args.bev_checkpoint, map_location='cpu')
        if 'meta' in checkpoint:
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            if 'PALETTE' in checkpoint['meta']:
                model.PALETTE = checkpoint['meta']['PALETTE']
        print(f"[get_bev_model] Loaded BEV checkpoint from {args.bev_checkpoint}")

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=False
        )
    return model



# from mmcv.parallel import scatter, DataContainer

# def unwrap_mmdet_batch(batch, device):
#     if isinstance(device, torch.device) and device.type == "cuda":
#         gpu_id = torch.cuda.current_device()
#         # scatter는 list를 반환하므로 0번째를 꺼냄
#         batch = scatter(batch, [gpu_id], dim=0)[0]
#         return batch
#     else:
#         # CPU일 경우 수동으로 DataContainer 풀기
#         def _unwrap(x):
#             if isinstance(x, DataContainer):
#                 return x.data[0]
#             elif isinstance(x, list):
#                 return [_unwrap(y) for y in x]
#             elif isinstance(x, dict):
#                 return {k: _unwrap(v) for k, v in x.items()}
#             else:
#                 return x
#         return _unwrap(batch)
    
from mmcv.parallel import scatter, DataContainer
import torch

def unwrap_mmdet_batch(batch, device):

    assert isinstance(device, torch.device) and device.type == "cuda", \
        "unwrap_mmdet_batch는 CUDA만 지원합니다."

    gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
    # scatter는 리스트/튜플을 반환 → 단일 타깃이라 [0]만 취함
    batch = scatter(batch, [gpu_idx], dim=0)[0]

    def _unwrap_dc(x):
        if isinstance(x, DataContainer):
            return _unwrap_dc(x.data[0])
        elif isinstance(x, dict):
            return {k: _unwrap_dc(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(_unwrap_dc(v) for v in x)
        else:
            return x

    return _unwrap_dc(batch)


from torch.nn.modules.batchnorm import _BatchNorm
class freeze_bn_stats:
    def __init__(self, model, accelerator):
        self.model = model
        self.accelerator = accelerator
        self._states = []
    def __enter__(self):
        base = self.accelerator.unwrap_model(self.model)
        self._states = []
        for m in base.modules():
            if isinstance(m, _BatchNorm):
                self._states.append((m, m.training))
                m.eval()  # running stats 업데이트 중지
        return self
    def __exit__(self, exc_type, exc, tb):
        for m, was_train in self._states:
            m.train(was_train)
        self._states = []
