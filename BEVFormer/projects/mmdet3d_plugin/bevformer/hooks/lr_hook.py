from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FixUnetLrHook(Hook):
    def __init__(self, unet_lr=1e-4):
        self.unet_lr = unet_lr

    def before_train_iter(self, runner):
        # optimizer의 param_groups를 순회하면서 이름 확인
        for i, group in enumerate(runner.optimizer.param_groups):
            # 'unet' 관련 그룹만 고정
            if 'pts_bbox_head.unet' in group.get('param_names', []):
                group['lr'] = self.unet_lr
