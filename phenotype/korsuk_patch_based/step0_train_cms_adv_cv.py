from __future__ import print_function
import os
import time
import torch
import argparse
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn
from dataset_TCGA import ImgDataset as ImgDataset_test
from dataset_imgaug_TCGA import ImgDataset as ImgDataset_train
import numpy as np
import deepdish
import torch.nn as nn
from torch.autograd import Variable
from inception import Inception
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import train_test_split

header = '/users/rittscher/korsuk/data'
# header = '/media/korsuks/rescomp'

parser = argparse.ArgumentParser()

parser.add_argument('--dir', default='x10', type=str)
parser.add_argument('--icv', default='cv0', type=str)

parser.add_argument('--iters', default=50001, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=0.0002, type=float)

parser.add_argument('--num_classes', default=4, type=int)
parser.add_argument('--cv_file', default='TCGA_cms_cv.h5', type=str)
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--checkpoint_folder', default='', type=str)

parser.add_argument('--resume', default='cms/x10/cv0/checkpoint/checkpoint_37500.pth', type=str)
parser.add_argument('--save_interval', default=500, type=int)

args = parser.parse_args()


class HLoss(nn.Module):
    def __init__(self, average=True):
        super(HLoss, self).__init__()
        self.average = average

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if self.average:
            b = -1.0 * b.mean()
        else:
            b = -1.0 * b.sum()

        return b


def create_dataset(dir, mode, cv):
    if mode == 'train':
        return ImgDataset_train(dir, mode, cv)
    else:
        return ImgDataset_test(dir, mode, cv)


if 'cv0' in args.resume:
    icv = 'cv0'
elif 'cv1' in args.resume:
    icv = 'cv1'
elif 'cv2' in args.resume:
    icv = 'cv2'
elif 'cv3' in args.resume:
    icv = 'cv3'
elif 'cv4' in args.resume:
    icv = 'cv4'
else:
    raise Exception('no cv')


def main():
    if args.dir == 'x5':
        args.dir = '../../../04_Intermediate/CNN/step2_denser_TCGA_cms_patches/x5'
        args.checkpoint_folder = 'cms/x5/'+ icv +'/checkpoint'
    elif args.dir == 'x10':
        args.dir = '../../../04_Intermediate/CNN/step2_denser_TCGA_cms_patches/x10'
        args.checkpoint_folder = 'cms/x10/'+ icv +'/checkpoint'
    else:
        args.dir = '../../../04_Intermediate/CNN/step2_denser_TCGA_cms_patches/x20'
        args.checkpoint_folder = 'cms/x20/'+ icv +'/checkpoint'

    # initialize CUDA
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cudnn.benchmark = True

    #######################################################################################################

    # create data loader
    cv = deepdish.io.load(args.cv_file)
    cv = cv[args.icv]

    cv['train']['img'], cv['val']['img'], cv['train']['label'], cv['val']['label'] = train_test_split(cv['train']['img'], cv['train']['label'], test_size=0.33, random_state=42)

    train_dataset = create_dataset(args.dir, 'train', cv)
    val_dataset = create_dataset(args.dir, 'val', cv)

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_dataset.weight, len(train_dataset.weight))
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_dataset.weight, len(val_dataset.weight))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.workers, sampler=train_sampler, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             num_workers=args.workers, sampler=val_sampler, pin_memory=True)

    # create model
    net = Inception(num_classes=args.num_classes)
    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    # create a checkpoint folder if not exists
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    # create an optimizer and loss criterion
    criterion = nn.CrossEntropyLoss()
    hloss = HLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    # optionally resume from a checkpoint
    start_iter = 0
    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []

    train_auc = []
    val_auc = []

    train_micro = []
    val_micro = []

    train_macro = []
    val_macro = []

    iter_array = []

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_iter = checkpoint['iter']

            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']

            try:
                train_acc = checkpoint['train_acc']
                val_acc = checkpoint['val_acc']

                train_auc = checkpoint['train_auc']
                val_auc = checkpoint['val_auc']

                train_micro = checkpoint['train_micro']
                val_micro = checkpoint['val_micro']

                train_macro = checkpoint['train_macro']
                val_macro = checkpoint['val_macro']
            except:
                pass

            iter_array = checkpoint['iter_array']

            print("=> loaded checkpoint '{}' (iteration {})"
                  .format(args.resume, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    t_loss_meter = AverageMeter()
    t_loss_meter.reset()

    v_loss_meter = AverageMeter()
    v_loss_meter.reset()

    t_micro_meter = AverageMeter()
    t_micro_meter.reset()

    t_macro_meter = AverageMeter()
    t_macro_meter.reset()

    v_micro_meter = AverageMeter()
    v_micro_meter.reset()

    v_macro_meter = AverageMeter()
    v_macro_meter.reset()

    t_auc_meter = AverageMeterList(args.num_classes)
    t_auc_meter.reset()

    v_auc_meter = AverageMeterList(args.num_classes)
    v_auc_meter.reset()

    t_acc_meter = AverageMeterList(args.num_classes)
    t_acc_meter.reset()

    v_acc_meter = AverageMeterList(args.num_classes)
    v_acc_meter.reset()

    iterable_train_loader = iter(train_loader)
    iterable_val_loader = iter(val_loader)

    y_train_label = []
    y_train_score = []

    y_test_label = []
    y_test_score = []
    for iteration in range(start_iter, args.iters):

        t = time.time()
        try:
            train_input, train_label = next(iterable_train_loader)
        except:
            iterable_train_loader = iter(train_loader)
            train_input, train_label = next(iterable_train_loader)

        train(train_input, train_label, net, optimizer, criterion, hloss, t_loss_meter, t_acc_meter, y_train_label, y_train_score)

        duration = time.time() - t

        # measure and record
        print_str = 'iteration: %d ' \
                    'train_loss: %.3f ' \
                    'train_time: %.3f'

        print(print_str % (iteration, t_loss_meter.avg, duration))

        #####################################################################
        with torch.no_grad():

            t = time.time()
            try:
                val_input, val_label = next(iterable_val_loader)
            except:
                iterable_val_loader = iter(val_loader)
                val_input, val_label = next(iterable_val_loader)

            validate(val_input, val_label, net, criterion, hloss, v_loss_meter, v_acc_meter, y_test_label, y_test_score)

            duration = time.time() - t

            # measure and record
            print_str = 'iteration: %d ' \
                        'val_loss: %.3f ' \
                        'val_time: %.3f'

            print(print_str % (iteration, v_loss_meter.avg, duration))

        if iteration % args.save_interval == 0 and iteration > 0:
            y_train_label = np.hstack(y_train_label)
            y_train_score = np.vstack(y_train_score)
            roc_auc = calculate_auc(y_train_label, y_train_score)
            t_micro_meter.update(roc_auc['micro'], y_train_label.shape[0])
            t_macro_meter.update(roc_auc['macro'], y_train_label.shape[0])
            t_auc_meter.update([roc_auc[i] for i in range(args.num_classes)], [(y_train_label == i).sum() for i in range(args.num_classes)])

            y_test_label = np.hstack(y_test_label)
            y_test_score = np.vstack(y_test_score)
            roc_auc = calculate_auc(y_test_label, y_test_score)
            v_micro_meter.update(roc_auc['micro'], y_test_label.shape[0])
            v_macro_meter.update(roc_auc['macro'], y_test_label.shape[0])
            v_auc_meter.update([roc_auc[i] for i in range(args.num_classes)],
                               [(y_test_label == i).sum() for i in range(args.num_classes)])

            # train and val losses
            train_loss.append(t_loss_meter.avg)
            val_loss.append(v_loss_meter.avg)

            train_acc.append(t_acc_meter.avg)
            val_acc.append(v_acc_meter.avg)

            train_micro.append(t_micro_meter.avg)
            val_micro.append(v_micro_meter.avg)

            train_macro.append(t_macro_meter.avg)
            val_macro.append(v_macro_meter.avg)

            train_auc.append(t_auc_meter.avg)
            val_auc.append(v_auc_meter.avg)

            iter_array.append(iteration)


            # save checkpoint
            save_name = os.path.join(args.checkpoint_folder,
                                     'checkpoint_' + str(iteration) + '.pth')
            save_variable_name = os.path.join(args.checkpoint_folder,
                                              'variables.h5')

            save_checkpoint({
                'iter': iteration + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'iter_array': iter_array,
                'train_micro':train_micro,
                'val_micro':val_micro,
                'train_macro':train_macro,
                'val_macro':val_macro,
                'train_auc':train_auc,
                'val_auc':val_auc},
                {'iter_array': np.array(iter_array),
                 'train_loss': np.array(train_loss),
                 'train_acc': np.vstack(train_acc),
                 'val_acc': np.vstack(val_acc),
                 'val_loss': np.array(val_loss),
                 'train_micro': np.array(train_micro),
                 'val_micro': np.array(val_micro),
                 'train_macro': np.array(train_macro),
                 'val_macro': np.array(val_macro),
                 'train_auc': np.array(train_auc),
                 'val_auc': np.array(val_auc)
                 },
                filename=save_name,
                variable_name=save_variable_name)

            # reset meter
            if iteration % args.save_interval == 0:
                t_loss_meter.reset()
                v_loss_meter.reset()

                t_acc_meter.reset()
                v_acc_meter.reset()

                t_micro_meter.reset()
                v_micro_meter.reset()

                t_macro_meter.reset()
                v_macro_meter.reset()

                t_auc_meter.reset()
                v_auc_meter.reset()

                y_train_label = []
                y_train_score = []

                y_test_label = []
                y_test_score = []


def train(input, label, net, optimizer, criterion, hloss, loss_meter, acc_meter, y_label, y_score):
    # switch to train mode
    net.train()

    input_var = Variable(input.cuda(async=True), requires_grad=True)
    label_var = label.cuda(async=True)

    # set the grad to zero
    optimizer.zero_grad()

    # run the model
    predict = net(input_var)

    # calculate loss
    if isinstance(predict, tuple):
        loss = sum((criterion(o, label_var) - hloss(o) for o in predict))
    else:
        loss = criterion(predict, label_var) - hloss(predict)

    # backward and optimizer
    loss.backward(retain_graph=True)

    # adversarial
    epsilon = 0.01*(input_var.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - input_var.min(dim=-1, keepdim=True)[0].min(dim=-1, keepdim=True)[0])
    x_grad = torch.sign(input_var.grad.data)
    adv_x = input_var.data + epsilon * x_grad  # we do not know the min/max because of torch's own stuff

    predict_adv = net(adv_x)
    # calculate loss
    if isinstance(predict_adv, tuple):
        loss_adv = sum((criterion(o, label_var) - hloss(o) for o in predict_adv))
    else:
        loss_adv = criterion(predict_adv, label_var) - hloss(predict_adv)

    all_loss = loss + loss_adv
    all_loss.backward()
    optimizer.step()

    # record loss
    loss_meter.update(all_loss.item(), len(input))

    # prediction acc
    if isinstance(predict, tuple):
        predict_score = F.softmax(predict[0], dim=1)
        predict = predict[0].argmax(dim=1)
    else:
        predict_score = F.softmax(predict, dim=1)
        predict = predict.argmax(dim=1)

    acc = [0] * args.num_classes
    n = [0] * args.num_classes
    predict = predict.cpu().data.numpy()

    label = label.cpu().data.numpy()

    for i in range(args.num_classes):
        n[i] = np.sum(label == i)
        acc[i] = np.logical_and(predict == i, label == i).sum()
        if n[i] == 0 and acc[i] == 0:
            acc[i] = 1.0
        elif n[i] == 0 and acc[i] > 0:
            acc[i] = 0.0
        else:
            acc[i] = acc[i] / n[i]

    acc_meter.update(acc, n)

    # # roc auc
    # roc_auc = calculate_auc(label_var.cpu().data.numpy(), predict_score.cpu().data.numpy())
    # micro_meter.update(roc_auc['micro'], len(input))
    # macro_meter.update(roc_auc['macro'], len(input))
    y_label.append(label)
    y_score.append(predict_score.cpu().data.numpy())


def validate(input, label, net, criterion, hloss, loss_meter, acc_meter, y_label, y_score):
    # switch to train mode
    net.eval()

    input_var = Variable(input.cuda(async=True), requires_grad=False)
    label_var = Variable(label.cuda(async=True), requires_grad=False)

    # run the model
    predict = net(input_var)

    # calculate loss
    if isinstance(predict, tuple):
        loss = sum((criterion(o, label_var) - hloss(o) for o in predict))
    else:
        loss = criterion(predict, label_var) - hloss(predict)

    # record loss
    loss_meter.update(loss.item(), input[0].size(0))

    # prediction acc
    if isinstance(predict, tuple):
        predict_score = F.softmax(predict[0], dim=1)
        predict = predict[0].argmax(dim=1)
    else:
        predict_score = F.softmax(predict, dim=1)
        predict = predict.argmax(dim=1)

    acc = [0] * args.num_classes
    n = [0] * args.num_classes
    predict = predict.cpu().data.numpy()
    label = label.cpu().data.numpy()

    for i in range(args.num_classes):
        n[i] = np.sum(label == i)
        acc[i] = np.logical_and(predict == i, label == i).sum()
        if n[i] == 0 and acc[i] == 0:
            acc[i] = 1.0
        elif n[i] == 0 and acc[i] > 0:
            acc[i] = 0.0
        else:
            acc[i] = acc[i] / n[i]

    acc_meter.update(acc, n)

    # # roc auc
    # roc_auc = calculate_auc(label_var.cpu().data.numpy(), predict_score.cpu().data.numpy())
    # micro_meter.update(roc_auc['micro'], len(input))
    # macro_meter.update(roc_auc['macro'], len(input))
    y_label.append(label)
    y_score.append(predict_score.cpu().data.numpy())


def save_checkpoint(state,
                    variables,
                    filename='checkpoint.pth.tar', variable_name='variables.h5'):
    torch.save(state, filename)
    deepdish.io.save(variable_name, variables)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterList(object):
    """Computes and stores the average and current value"""

    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.reset()

    def reset(self):
        self.val = [0] * self.nclasses
        self.avg = [0] * self.nclasses
        self.sum = [0] * self.nclasses
        self.count = [0] * self.nclasses

    def update(self, val, n):
        for i in range(self.nclasses):
            if n[i] > 0:
                self.val[i] = val[i]
                self.sum[i] += val[i] * n[i]
                self.count[i] += n[i]
                self.avg[i] = self.sum[i] / self.count[i]

def calculate_auc(y_test, y_score):

    onehot_encoder = OneHotEncoder(categories=[range(args.num_classes)], sparse=False)
    integer_encoded = y_test.reshape(len(y_test), 1)
    y_test = onehot_encoder.fit_transform(integer_encoded)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    num_classes = 0
    for i in range(args.num_classes):
        if np.any(np.isnan(tpr[i])):
            continue
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        num_classes += 1

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc


if __name__ == '__main__':
    main()
