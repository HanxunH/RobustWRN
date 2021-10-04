import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy as np
from torch.autograd import Variable
import util
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class MadrysLoss(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=6.0,
                 distance='l_inf', cutmix=False, adjust_freeze=True, cutout=False,
                 cutout_length=16):
        super(MadrysLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance
        self.cross_entropy = models.CutMixCrossEntropyLoss() if cutmix else torch.nn.CrossEntropyLoss()
        self.adjust_freeze = adjust_freeze
        self.cutout = cutout
        self.cutout_length = cutout_length

    def forward(self, model, x_natural, labels, optimizer):
        model.eval()
        if self.adjust_freeze:
            for param in model.parameters():
                param.requires_grad = False

        # generate adversarial example
        x_adv = x_natural.detach() + self.step_size * torch.randn(x_natural.shape).to(device).detach()
        if self.distance == 'l_inf':
            adv_loss = 0
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), labels)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = Variable(x_adv, requires_grad=False)

        if self.adjust_freeze:
            for param in model.parameters():
                param.requires_grad = True

        if self.cutout:
            batch_size = x_adv.shape[0]
            c, h, w = x_adv.shape[1], x_adv.shape[2], x_adv.shape[3]
            mask = torch.ones(batch_size, c, h, w).float()
            for j in range(batch_size):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.cutout_length // 2, 0, h)
                y2 = np.clip(y + self.cutout_length // 2, 0, h)
                x1 = np.clip(x - self.cutout_length // 2, 0, w)
                x2 = np.clip(x + self.cutout_length // 2, 0, w)

                mask[j, :, y1: y2, x1: x2] = 0.0
            x_adv = x_adv * mask.to(device)

        model.train()
        optimizer.zero_grad()

        if isinstance(model, nn.DataParallel):
            check_model = model.module
        else:
            check_model = model

        if isinstance(check_model, models.DARTS_model.NetworkCIFAR) and check_model._auxiliary:
            logits, aux_logits = model(x_adv)
            loss = self.cross_entropy(logits, labels) + check_model.aux_weights * self.cross_entropy(aux_logits, labels)
        elif isinstance(check_model, models.basic_model.RobNetwork) and check_model._auxiliary:
            logits, aux_logits = model(x_adv)
            loss = self.cross_entropy(logits, labels) + check_model.aux_weights * self.cross_entropy(aux_logits, labels)
        else:
            logits = model(x_adv)
            loss = self.cross_entropy(logits, labels)

        return logits, loss
