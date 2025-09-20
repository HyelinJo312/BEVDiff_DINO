from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class DiffusionCurriculumHook(Hook):
    def __init__(self, head_attr_path='pts_bbox_head', total_epochs=None):
        self.head_attr_path = head_attr_path
        self.cfg_total_epochs = total_epochs  # runner.max_epochs가 없을 때 fallback

    def before_train_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):  # DDP unwrap
            model = model.module

        # 모델 안에서 head까지 타고 들어가기 (ex: 'pts_bbox_head' or 'bbox_head.xxx')
        head = model
        for name in self.head_attr_path.split('.'):
            if not hasattr(head, name):
                # 경로가 틀렸다면 그냥 무시하고 리턴
                return
            head = getattr(head, name)

        # BEVDiffuserHead.set_epoch(curr, total) 호출
        if hasattr(head, 'set_epoch'):
            total = getattr(runner, 'max_epochs', None)
            if total is None:
                total = self.cfg_total_epochs  # 마지막 안전장치
            head.set_epoch(runner.epoch, total)
