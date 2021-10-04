import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import json
from torch.autograd import Variable

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def local_lip(model, x, xp, top_norm=1, btm_norm=float('inf'), reduction='mean'):
    model.eval()
    down = torch.flatten(x - xp, start_dim=1)
    with torch.no_grad():
        if top_norm == "kl":
            criterion_kl = nn.KLDivLoss(reduction='none')
            top = criterion_kl(F.log_softmax(model(xp), dim=1),
                               F.softmax(model(x), dim=1))
            ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
        else:
            top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
            ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

    if reduction == 'mean':
        return torch.mean(ret)
    elif reduction == 'sum':
        return torch.sum(ret)
    else:
        raise ValueError("Not supported reduction")


def fosc(model, x, x_adv, labels, epsilon):
    batch_size = x_adv.shape[0]

    model.zero_grad()
    x_adv = Variable(x_adv, requires_grad=True)
    adv_logits = model(x_adv)
    loss = F.cross_entropy(adv_logits, labels, reduction='none')
    loss.sum().backward()
    grad = x_adv.grad.data

    grad_flatten = grad.view(grad.shape[0], -1)
    grad_norm = torch.norm(grad_flatten, 1, dim=1)
    x_diff = (x_adv.detach() - x).view(batch_size, -1)
    fosc_value = epsilon * grad_norm - (grad_flatten * x_diff).sum(dim=1)
    return fosc_value.view(batch_size)


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = '\t' + key + '=' + value
        else:
            display += '\t' + str(key) + '=%.4f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
    return res


def save_model(filename, epoch, model, optimizer, alpha_optimizer, scheduler, save_best=False, **kwargs):
    # Torch Save State Dict
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'alpha_optimizer_state_dict': alpha_optimizer.state_dict() if alpha_optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, filename)
    if save_best:
        pre, ext = os.path.splitext(filename)
        pre += '_best'
        filename = pre + ext
        torch.save(state, filename)
    return


def load_model(filename, model, optimizer, alpha_optimizer,  scheduler, **kwargs):
    # Load Torch State Dict
    checkpoints = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoints['model_state_dict'], strict=False)
    if optimizer is not None and checkpoints['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    if alpha_optimizer is not None and checkpoints['alpha_optimizer_state_dict'] is not None:
        alpha_optimizer.load_state_dict(checkpoints['alpha_optimizer_state_dict'])
    if scheduler is not None and checkpoints['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    return checkpoints


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            n = nn.init._calculate_correct_fan(m.popup_scores, "fan_in")
            # Close to kaiming unifrom init
            m.popup_scores.data = (
                math.sqrt(6 / n) * m.weight.data / torch.max(torch.abs(m.weight.data))
            )


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
