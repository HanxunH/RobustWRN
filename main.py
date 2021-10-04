import mlconfig
import argparse
import datetime
import util
import models
import dataset
import trades
import madrys
import mart
import time
import os
import torch
import shutil
import numpy as np
from trainer import Trainer
from evaluator import Evaluator
from torch.autograd import Variable
from auto_attack.autoattack import AutoAttack
from thop import profile

mlconfig.register(trades.TradesLoss)
mlconfig.register(madrys.MadrysLoss)
mlconfig.register(mart.MartLoss)
mlconfig.register(dataset.DatasetGenerator)

parser = argparse.ArgumentParser(description='RobustArc')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--version', type=str, default="DARTS_Search")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_best_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--attack_choice', default='PGD', choices=['PGD', 'AA', 'GAMA', 'CW', 'none'])
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--num_steps', default=20, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')
parser.add_argument('--train_eval_epoch', default=0.5, type=float, help='PGD Eval in training after this epoch')
args = parser.parse_args()
if args.epsilon > 1:
    args.epsilon = args.epsilon / 255
    args.step_size = args.step_size / 255

# Set up
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now()

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
search_results_checkpoint_file_name = None

checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")


torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


def whitebox_eval(data_loader, model, evaluator, log=True):
    natural_count, pgd_count, total, stable_count = 0, 0, 0, 0
    loss_meters = util.AverageMeter()
    lip_meters = util.AverageMeter()

    model.eval()
    for i, (images, labels) in enumerate(data_loader["test_dataset"]):
        images, labels = images.to(device), labels.to(device)
        # pgd attack
        images, labels = Variable(images, requires_grad=True), Variable(labels)
        if args.attack_choice == 'PGD':
            rs = evaluator._pgd_whitebox(model, images, labels, random_start=True,
                                         epsilon=args.epsilon, num_steps=args.num_steps,
                                         step_size=args.step_size)
        elif args.attack_choice == 'GAMA':
            rs = evaluator._gama_whitebox(model, images, labels, epsilon=args.epsilon,
                                          num_steps=args.num_steps, eps_iter=args.step_size)
        elif args.attack_choice == 'CW':
            rs = evaluator._cw_whitebox(model, images, labels, random_start=True,
                                        epsilon=args.epsilon, num_steps=args.num_steps,
                                        step_size=args.step_size)
        else:
            raise('Not implemented')
        acc, acc_pgd, loss, stable, X_pgd = rs
        total += images.size(0)
        natural_count += acc
        pgd_count += acc_pgd
        stable_count += stable
        local_lip = util.local_lip(model, images, X_pgd).item()
        lip_meters.update(local_lip)
        loss_meters.update(loss)
        if log:
            payload = 'LIP: %.4f\tStable Count: %d\tNatural Count: %d/%d\tNatural Acc: %.2f\tAdv Count: %d/%d\tAdv Acc: %.2f' % (local_lip, stable_count, natural_count, total, (natural_count/total) * 100, pgd_count, total, (pgd_count/total) * 100)
            logger.info(payload)

    natural_acc = (natural_count/total) * 100
    pgd_acc = (pgd_count/total) * 100
    payload = 'Natural Correct Count: %d/%d Acc: %.2f ' % (natural_count, total, natural_acc)
    logger.info(payload)
    payload = '%s Correct Count: %d/%d Acc: %.2f ' % (args.attack_choice, pgd_count, total, pgd_acc)
    logger.info(payload)
    payload = '%s Loss Avg: %.2f ' % (args.attack_choice, loss_meters.avg)
    logger.info(payload)
    payload = 'LIP Avg: %.4f ' % (lip_meters.avg)
    logger.info(payload)
    payload = 'Stable Count: %d/%d StableAcc: %.2f ' % (stable_count, total, stable_count * 100/total)
    logger.info(payload)
    return natural_acc, pgd_acc, stable_count*100/total, lip_meters.avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = config.optimizer.lr
    schedule = config.lr_schedule if hasattr(config, 'lr_schedule') else 'fixed'
    if schedule == 'fixed':
        if epoch >= 0.75 * config.epochs:
            lr = config.optimizer.lr * 0.1
        if epoch >= 0.9 * config.epochs:
            lr = config.optimizer.lr * 0.01
        if epoch >= config.epochs:
            lr = config.optimizer.lr * 0.001
    # cosine schedule
    elif schedule == 'cosine':
        lr = config.optimizer.lr * 0.5 * (1 + np.cos((epoch - 1) / config.epochs * np.pi))
    elif schedule == 'search':
        if epoch >= 75:
            lr = 0.01
        if epoch >= 90:
            lr = 0.001
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_eps(epoch, config):
    eps_min = 2/255
    eps_max = 8/255
    ratio = epoch / (config.epochs * 0.2)
    eps = (eps_min + 0.5 * (eps_max - eps_min) * (1 - np.cos(ratio * np.pi)))
    return eps


def adjust_weight_decay(model, l2_value):
    conv, fc = [], []
    for name, param in model.named_parameters():
        print(name)
        if not param.requires_grad:
            # frozen weights
            continue
        if 'fc' in name:
            fc.append(param)
        else:
            conv.append(param)
    params = [{'params': conv, 'weight_decay': l2_value}, {'params': fc, 'weight_decay': 0.01}]
    print(fc)
    return params


def train(starting_epoch, model, genotype, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    print(model)
    for epoch in range(starting_epoch, config.epochs):
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        adjust_learning_rate(optimizer, epoch)

        # Update Drop Path Prob
        if isinstance(model, models.DARTS_model.NetworkCIFAR):
            drop_path_prob = config.model.drop_path_prob * epoch / config.epochs
            model.drop_path_prob = drop_path_prob
            logger.info('Drop Path Probability %.4f' % drop_path_prob)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        evaluator.eval(epoch, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100

        is_best = False
        if epoch >= config.epochs * args.train_eval_epoch and args.train_eval_epoch >= 0:
            # Reset Stats
            trainer._reset_stats()
            evaluator._reset_stats()
            for param in model.parameters():
                param.requires_grad = False
            natural_acc, pgd_acc, stable_acc, lip = whitebox_eval(data_loader, model, evaluator, log=False)
            for param in model.parameters():
                param.requires_grad = True
            is_best = True if pgd_acc > ENV['best_pgd_acc'] else False
            ENV['best_pgd_acc'] = max(ENV['best_pgd_acc'], pgd_acc)
            ENV['pgd_eval_history'].append((epoch, pgd_acc))
            ENV['stable_acc_history'].append(stable_acc)
            ENV['lip_history'].append(lip)
            logger.info('Best PGD accuracy: %.2f' % (ENV['best_pgd_acc']))
        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        filename = checkpoint_path_file + '.pth'
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        alpha_optimizer=None,
                        scheduler=None,
                        genotype=genotype,
                        save_best=is_best,
                        filename=filename)
        logger.info('Model Saved at %s\n', filename)
    return


def main():
    # Load Search Version Genotype
    model = config.model().to(device)
    genotype = None

    # Setup ENV
    data_loader = config.dataset().getDataLoader()
    if hasattr(config, 'adjust_weight_decay') and config.adjust_weight_decay:
        params = adjust_weight_decay(model, config.optimizer.weight_decay)
    else:
        params = model.parameters()
    optimizer = config.optimizer(params)
    criterion = config.criterion()
    trainer = Trainer(criterion, data_loader, logger, config)
    evaluator = Evaluator(data_loader, logger, config)
    profile_inputs = (torch.randn([1, 3, 32, 32]).to(device),)
    flops, params = profile(model, inputs=profile_inputs, verbose=False)
    flops = flops / 1e6
    starting_epoch = 0

    config.set_immutable()
    for key in config:
        logger.info("%s: %s" % (key, config[key]))
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    logger.info("flops: %.4fM" % flops)
    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'flops': flops,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'stable_acc_history': [],
           'lip_history': [],
           'genotype_list': []}

    if args.load_model or args.load_best_model:
        filename = checkpoint_path_file + '_best.pth' if args.load_best_model else checkpoint_path_file + '.pth'
        checkpoint = util.load_model(filename=filename,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=None)
        starting_epoch = checkpoint['epoch'] + 1
        ENV = checkpoint['ENV']
        if 'stable_acc_history' not in ENV:
            ENV['stable_acc_history'] = []
        if 'lip_history' not in ENV:
            ENV['lip_history'] = []
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (filename))

    if args.data_parallel:
        print('data_parallel')
        model = torch.nn.DataParallel(model).to(device)

    logger.info("Starting Epoch: %d" % (starting_epoch))

    if args.train:
        train(starting_epoch, model, genotype, optimizer, None, criterion, trainer, evaluator, ENV, data_loader)
    elif args.attack_choice in ['PGD', 'GAMA', 'CW']:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        natural_acc, adv_acc, stable_acc, lip = whitebox_eval(data_loader, model, evaluator)
        key = '%s_%d' % (args.attack_choice, args.num_steps)
        ENV['natural_acc'] = natural_acc
        ENV[key] = adv_acc
        ENV['%s_stable' % key] = stable_acc
        ENV['%s_lip' % key] = lip
        target_model = model.module if args.data_parallel else model
        filename = checkpoint_path_file + '_best.pth' if args.load_best_model else checkpoint_path_file + '.pth'
        util.save_model(ENV=ENV,
                        epoch=starting_epoch-1,
                        model=target_model,
                        optimizer=optimizer,
                        alpha_optimizer=None,
                        scheduler=None,
                        genotype=genotype,
                        filename=filename)
    elif args.attack_choice == 'AA':
        for param in model.parameters():
            param.requires_grad = False
        x_test = [x for (x, y) in data_loader['test_dataset']]
        x_test = torch.cat(x_test, dim=0)
        y_test = [y for (x, y) in data_loader['test_dataset']]
        y_test = torch.cat(y_test, dim=0)
        model.eval()

        adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, logger=logger, verbose=True)
        adversary.plus = False

        logger.info('=' * 20 + 'AA Attack Eval' + '=' * 20)
        x_adv, robust_accuracy = adversary.run_standard_evaluation(x_test, y_test, bs=config.dataset.eval_batch_size)
        robust_accuracy = robust_accuracy * 100
        logger.info('AA Accuracy: %.2f' % (robust_accuracy))

        ENV['aa_attack'] = robust_accuracy

        target_model = model.module if args.data_parallel else model
        filename = checkpoint_path_file + '_best.pth' if args.load_best_model else checkpoint_path_file + '.pth'
        util.save_model(ENV=ENV,
                        epoch=starting_epoch-1,
                        model=target_model,
                        optimizer=optimizer,
                        alpha_optimizer=None,
                        scheduler=None,
                        genotype=genotype,
                        filename=filename)
    return


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)
