import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import models
import util
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


class TradesLoss(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=6.0,
                 distance='l_inf', ce=False, cutmix=False, adjust_freeze=True):
        super(TradesLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.distance = distance
        self.ce = ce
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.cross_entropy = models.CutMixCrossEntropyLoss() if cutmix else torch.nn.CrossEntropyLoss()
        self.adjust_freeze = adjust_freeze

    def forward(self, model, x_natural, y, optimizer):
        # define KL-loss
        criterion_kl = self.criterion_kl
        model.eval()
        if self.adjust_freeze:
            for param in model.parameters():
                param.requires_grad = False

        # generate adversarial example
        batch_size = len(x_natural)
        logits = model(x_natural)
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                if self.ce:
                    loss_kl = self.cross_entropy(model(x_adv), y)
                else:
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(logits, dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif self.distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=self.epsilon / self.perturb_steps * 2)

            for _ in range(self.perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(logits, dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if self.adjust_freeze:
            for param in model.parameters():
                param.requires_grad = True

        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits = model(x_natural)
        loss_natural = self.cross_entropy(logits, y)
        adv_logits = model(x_adv)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                        F.softmax(logits, dim=1))

        loss = loss_natural + self.beta * loss_robust
        return adv_logits, loss
