# projects/logger.py  (또는 projects/logger/compact_text_logger.py)
from mmcv.runner import TextLoggerHook, HOOKS, Hook

# @HOOKS.register_module()
# class CompactTextLoggerHook(TextLoggerHook):
#     # 원하는 키만 남기세요
#     KEEP_KEYS = {
#         'loss', 'loss_cls', 'loss_bbox', 'loss_diff',
#         'grad_norm', 'lr', 'time', 'data_time', 'memory', 'eta'
#     }
#     # _log_info가 필요로 하는 메타 키(절대 지우지 말 것)
#     REQUIRED_META = {'mode', 'epoch', 'iter'}

#     def log(self, runner):
#         # 1) 원본 로그 가져오기 (버전차이 안전하게 처리)
#         try:
#             log_dict = dict(runner.log_buffer.output)
#         except Exception:
#             log_dict = runner.log_buffer.output.copy() if hasattr(runner.log_buffer, 'output') else {}

#         # 2) 메타 키 보강
#         if 'mode' not in log_dict:
#             log_dict['mode'] = getattr(runner, 'mode', 'train')
#         if 'epoch' not in log_dict:
#             log_dict['epoch'] = getattr(runner, 'epoch', 0)
#         if 'iter' not in log_dict:
#             log_dict['iter'] = getattr(runner, 'iter', 0)

#         # 3) lr가 없으면 runner에서 가져와서 보강
#         if 'lr' not in log_dict:
#             try:
#                 lrs = runner.current_lr()
#                 if isinstance(lrs, (list, tuple)):
#                     lr_val = sum(lrs) / len(lrs) if lrs else 0.0
#                 elif isinstance(lrs, dict):
#                     lr_val = list(lrs.values())[0]
#                 else:
#                     lr_val = float(lrs)
#                 log_dict['lr'] = lr_val
#             except Exception:
#                 pass  # 못 가져와도 무시

#         if 'memory' not in log_dict:
#             mem_mb = 0.0
#             try:
#                 import torch
#                 if torch.cuda.is_available():
#                     mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
#             except Exception:
#                 pass
#             log_dict['memory'] = mem_mb


#         # 4) 키 필터링 (메타 키 + KEEP_KEYS + val/* 유지)
#         filtered = {}
#         for k, v in log_dict.items():
#             if (k in self.REQUIRED_META) or (k in self.KEEP_KEYS) or (isinstance(k, str) and k.startswith('val/')):
#                 filtered[k] = v

#         # 5) 키 이름 바꾸기 (옵션)
#         # if 'grad_norm' in filtered:
#         #     filtered['gnorm'] = filtered.pop('grad_norm')

#         # 6) 소수 자리 정리 (옵션)
#         for k, v in list(filtered.items()):
#             if isinstance(v, float):
#                 filtered[k] = float(f'{v:.4f}')

#         # 7) 부모 포맷터로 출력 (Epoch[..] lr:.. 형태 유지)
#         self._log_info(filtered, runner)




# @HOOKS.register_module()
# class FillMissingTimeHook(Hook):
#     """Ensure 'time' and 'data_time' exist before TextLoggerHook logs."""
#     def after_train_iter(self, runner):
#         out = runner.log_buffer.output
#         if 'time' not in out or 'data_time' not in out:
#             runner.log_buffer.update({'time': 0.0, 'data_time': 0.0}, count=1)


# 원본 함수 백업
_ORIG__log_info = TextLoggerHook._log_info

def _SAFE__log_info(self, log_dict, runner):
    # mode 기본값 보장
    mode = log_dict.get('mode', 'train')
    if mode == 'train':
        # train 포맷에서는 time/data_time이 없어도 기본값으로 채움
        log_dict.setdefault('time', 0.0)
        log_dict.setdefault('data_time', 0.0)
    # 그대로 원본 로직 실행
    return _ORIG__log_info(self, log_dict, runner)

# 몽키패치 적용
TextLoggerHook._log_info = _SAFE__log_info