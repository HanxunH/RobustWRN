import models
import util
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Evaluator():
    def __init__(self, data_loader, logger, config):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        return

    def eval(self, epoch, model):
        model.eval()
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels, model=model)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        return

    def eval_batch(self, images, labels, model):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(images)
            loss = self.criterion(pred, labels)
            acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
        self.loss_meters.update(loss.item(), n=images.size(0))
        self.acc_meters.update(acc.item(), n=images.size(0))
        self.acc5_meters.update(acc5.item(), n=images.size(0))
        payload = {"val_acc": acc.item(),
                   "val_acc_avg": self.acc_meters.avg,
                   "val_acc5": acc5.item(),
                   "val_acc5_avg": self.acc5_meters.avg,
                   "val_loss": loss.item(),
                   "val_loss_avg": self.loss_meters.avg}
        return payload

    def _pgd_whitebox(self, model, X, y, random_start=True,
                      epsilon=0.031, num_steps=20, step_size=0.003):
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd = Variable(X_pgd.data, requires_grad=False)
        predict_pgd = model(X_pgd).data.max(1)[1].detach()
        predict_clean = model(X).data.max(1)[1].detach()
        acc_pgd = (predict_pgd == y.data).float().sum()
        stable = (predict_pgd.data == predict_clean.data).float().sum()
        return acc.item(), acc_pgd.item(), loss.item(), stable.item(), X_pgd

    def _cw_whitebox(self, model, X, y, random_start=True,
                     epsilon=0.031, num_steps=20, step_size=0.003):
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)

        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                correct_logit = torch.sum(torch.gather(model(X_pgd), 1, (y.unsqueeze(1)).long()).squeeze())
                tmp1 = torch.argsort(model(X_pgd), dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                wrong_logit = torch.sum(torch.gather(model(X_pgd), 1, (new_y.unsqueeze(1)).long()).squeeze())
                loss = - F.relu(correct_logit-wrong_logit)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd = Variable(X_pgd.data, requires_grad=False)
        predict_pgd = model(X_pgd).data.max(1)[1].detach()
        predict_clean = model(X).data.max(1)[1].detach()
        acc_pgd = (predict_pgd == y.data).float().sum()
        stable = (predict_pgd.data == predict_clean.data).float().sum()

        return acc.item(), acc_pgd.item(), loss.item(), stable.item(), X_pgd

    def _gama_whitebox(self, model, X, y, epsilon=0.031, num_steps=100, eps_iter=0.0627,
                       bounds=np.array([[0, 1], [0, 1], [0, 1]]), w_reg=50, lin=25, SCHED=[60, 85],
                       drop=10):

        # Margin Loss
        def max_margin_loss(x, y):
            B = y.size(0)
            corr = x[range(B), y]

            x_new = x - 1000 * torch.eye(x.shape[1])[y].to(device)
            tar = x[range(B), x_new.argmax(dim=1)]
            loss = tar - corr
            loss = torch.mean(loss)

            return loss

        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()

        # PGD GAMA
        W_REG = w_reg
        B, C, H, W = X.size()
        noise = torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).to(device)
        noise = epsilon * torch.sign(noise)
        noise_X = Variable(X.data + noise, requires_grad=True)

        for step in range(num_steps):
            X_pgd = Variable(X.data + noise, requires_grad=True)
            if step in SCHED:
                eps_iter /= drop

            # forward pass
            orig_out = model(noise_X)
            P_out = nn.Softmax(dim=1)(orig_out)

            out = model(X_pgd)
            Q_out = nn.Softmax(dim=1)(out)

            # compute loss using true label
            if step <= lin:
                cost = W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + max_margin_loss(Q_out, y)
                W_REG -= w_reg/lin
            else:
                cost = max_margin_loss(Q_out, y)

            # backward pass
            cost.backward()
            # get gradient of loss wrt data
            per = torch.sign(X_pgd.grad.data)
            # convert eps 0-1 range to per channel range
            per[:, 0, :, :] = (eps_iter * (bounds[0, 1] - bounds[0, 0])) * per[:, 0, :, :]
            if(per.size(1) > 1):
                per[:, 1, :, :] = (eps_iter * (bounds[1, 1] - bounds[1, 0])) * per[:, 1, :, :]
                per[:, 2, :, :] = (eps_iter * (bounds[2, 1] - bounds[2, 0])) * per[:, 2, :, :]

            # ascent
            adv = X_pgd.data + per.to(device)

            # clip per channel data out of the range
            X_pgd.requires_grad = False
            X_pgd[:, 0, :, :] = torch.clamp(adv[:, 0, :, :], bounds[0, 0], bounds[0, 1])
            if(per.size(1) > 1):
                X_pgd[:, 1, :, :] = torch.clamp(adv[:, 1, :, :], bounds[1, 0], bounds[1, 1])
                X_pgd[:, 2, :, :] = torch.clamp(adv[:, 2, :, :], bounds[2, 0], bounds[2, 1])
            X_pgd = X_pgd.data
            noise = X_pgd - X
            noise = torch.clamp(noise, -epsilon, epsilon)

        predict_pgd = model(X_pgd).data.max(1)[1].detach()
        predict_clean = model(X).data.max(1)[1].detach()
        acc_pgd = (predict_pgd == y.data).float().sum()
        stable = (predict_pgd.data == predict_clean.data).float().sum()
        return acc.item(), acc_pgd.item(), cost.item(), stable.item(), X_pgd
