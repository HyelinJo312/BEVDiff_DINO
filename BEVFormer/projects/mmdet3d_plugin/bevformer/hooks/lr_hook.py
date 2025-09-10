from mmcv.runner import HOOKS, Hook

# @HOOKS.register_module()
# class FixUnetLrHook(Hook):
#     def __init__(self, unet_lr=1e-4):
#         self.unet_lr = unet_lr

#     def before_train_iter(self, runner):
#         # optimizer의 param_groups를 순회하면서 이름 확인
#         for i, group in enumerate(runner.optimizer.param_groups):
#             # 'unet' 관련 그룹만 고정
#             if 'pts_bbox_head.unet' in group.get('param_names', []):
#                 group['lr'] = self.unet_lr

@HOOKS.register_module()
class FixUnetLrHook(Hook):
    def __init__(self, unet_lr=1e-4, unet_attr_path='pts_bbox_head.unet', print_every=5000):
        priority = 'VERY_LOW'
        self.unet_lr = unet_lr
        self.unet_attr_path = unet_attr_path
        self._unet_param_ids = None
        self._unet_group_indices = None
        self.print_every = print_every

    def _resolve_attr(self, model, path: str):
        def walk(base, p):
            cur = base
            for name in p.split('.'):
                cur = getattr(cur, name)
            return cur

        candidates = []
      
        candidates.append((model, path))

        if path.startswith('module.'):
            candidates.append((model, path[len('module.'):]))
        else:
            candidates.append((model, 'module.' + path))

        if hasattr(model, 'module'):
            candidates.append((model.module, path))
            if path.startswith('module.'):
                candidates.append((model.module, path[len('module.'):]))
            else:
                candidates.append((model.module, 'module.' + path))

        last_err = None
        for base, p in candidates:
            try:
                return walk(base, p)
            except AttributeError as e:
                last_err = e
        raise AttributeError(f"[FixUnetLrHook] Cannot resolve attr path '{path}' on model.") from last_err

    def before_run(self, runner):
        unet = self._resolve_attr(runner.model, self.unet_attr_path)
        unet_params = list(unet.parameters())
        self._unet_param_ids = set(id(p) for p in unet_params)

        self._unet_group_indices = []
        for gi, g in enumerate(runner.optimizer.param_groups):
            params = g['params']
            if any(id(p) in self._unet_param_ids for p in params):
                self._unet_group_indices.append(gi)

        if len(self._unet_group_indices) == 0:
            runner.logger.warning(
                '[FixUnetLrHook] No optimizer param_group matched UNet params. '
                'Check unet_attr_path or your model structure.'
            )
        # runner.logger.info(f'[FixUnetLrHook] matched groups: {self._unet_group_indices}')

    def before_train_iter(self, runner):
        if not self._unet_group_indices:
            return
        for gi in self._unet_group_indices:
            runner.optimizer.param_groups[gi]['lr'] = self.unet_lr
            
            
    def before_train_iter(self, runner):
        if not self._unet_group_indices:
            return
        for gi in self._unet_group_indices:
            g = runner.optimizer.param_groups[gi]   
            g['lr'] = self.unet_lr                  
            if 'initial_lr' in g:                 
                g['initial_lr'] = self.unet_lr
        if getattr(self, 'print_every', 0) and (runner.iter % self.print_every == 0):
            runner.logger.info(f'[FixUnetLrHook] (iter={runner.iter}) force UNet lr={self.unet_lr}')

            
    def before_train_epoch(self, runner):
        if not self._unet_group_indices:
            return
        for gi in self._unet_group_indices:
            g = runner.optimizer.param_groups[gi]
            g['lr'] = self.unet_lr
            if 'initial_lr' in g:
                g['initial_lr'] = self.unet_lr
        runner.logger.info(f'[FixUnetLrHook] (epoch={runner.epoch}) force UNet lr={self.unet_lr:.3e}')
