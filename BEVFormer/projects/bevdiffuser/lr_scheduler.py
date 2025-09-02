import torch
import torch.nn.functional as F
import math

def get_cosine_annealing_with_warmup_scheduler(optimizer, total_iters, warmup_iters=500, warmup_ratio=1.0/3, min_lr_ratio=1e-3):
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            # Linear warmup from warmup_ratio to 1.0
            return warmup_ratio + (1.0 - warmup_ratio) * (current_iter / warmup_iters)
        else:
            # Cosine annealing from 1.0 to min_lr_ratio
            progress = (current_iter - warmup_iters) / (total_iters - warmup_iters)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
    

def get_bev_cosine_warmup_lambda(total_iters, warmup_iters=500, warmup_ratio=1.0/3, min_lr_ratio=1e-3):
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            # Linear warmup from warmup_ratio to 1.0
            return warmup_ratio + (1.0 - warmup_ratio) * (current_iter / float(max(1, warmup_iters)))
        # Cosine annealing from 1.0 to min_lr_ratio
        progress = (current_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return lr_lambda

    
def get_cyclic_scheduler(optimizer, total_steps, warmup_steps, max_lr=2e-4, min_lr_ratio=1e-3):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: 0 → 1.0
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay: 1.0 → min_lr_ratio
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
    

def get_step_scheduler(optimizer, drop_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < drop_epochs[0]:
            return 1.0
        elif current_epoch < drop_epochs[1]:
            return 0.1
        else:
            return 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def get_diffusers_lambda(name, total_steps, warmup_steps=0, *,
                          num_cycles=0.5, power=1.0, min_lr_ratio=0.0):
    """
    diffusers.get_scheduler(name, ...)와 동일한 러닝레이트 배율 곡선을 LambdaLR로 재현
    반환값: step -> multiplier (기본 lr에 곱해지는 배율)
    """
    def linear(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)  # 최종 0으로 감쇠

    def cosine(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # diffusers 기본: 0.5*(1+cos(pi*progress)) 에 min_lr_ratio 적용
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos_decay

    def cosine_with_restarts(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # num_cycles 번 재시작
        if num_cycles <= 0:
            cycles = 1.0
        else:
            cycles = num_cycles
        phi = math.pi * ((progress * cycles) % 1.0)
        cos_decay = 0.5 * (1.0 + math.cos(phi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos_decay

    def polynomial(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # (1 - progress) ** power (최종 0으로 감쇠)
        return max(0.0, (1.0 - progress) ** power)

    def constant(step):
        return 1.0

    def constant_with_warmup(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return 1.0

    name = name.lower()
    if name in ["linear"]:
        return linear
    elif name in ["cosine"]:
        # 필요시 args에 min_lr_ratio(=0.0)나 num_cycles(=0.5) 옵션 추가 가능
        return lambda s: cosine(s)
    elif name in ["cosine_with_restarts"]:
        return lambda s: cosine_with_restarts(s)
    elif name in ["polynomial"]:
        return lambda s: polynomial(s)
    elif name in ["constant"]:
        return constant
    elif name in ["constant_with_warmup"]:
        return constant_with_warmup
    else:
        raise ValueError(f"Unsupported scheduler name for LambdaLR: {name}")