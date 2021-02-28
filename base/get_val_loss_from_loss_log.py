from pathlib import Path
from argparse import ArgumentParser
import re
import json
from matplotlib import pyplot as plt
from base.utils.utils import AverageMeter


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('loss_log', type=Path)
    args = parser.parse_args()
    epochs, bces, accs, dices = [], [], [], []
    bces_avg, accs_avg, dices_avg = [], [], []
    bce_meter, acc_meter, dice_meter = AverageMeter(), AverageMeter(), AverageMeter()
    with args.loss_log.open('r') as loss_log_file:
        for line in loss_log_file:
            if 'validation' in line:
                epoch, bce, acc, dice = re.search(
                    r'\(epoch: (\d*), validation\) bce: (\d*.?\d*) acc: (\d*.?\d*) dice: (\d*.?\d*)',
                    line).groups()
                bce_meter.update(float(bce))
                acc_meter.update(float(acc))
                dice_meter.update(float(dice))
                bces_avg.append(float(bce_meter.avg)), accs_avg.append(float(acc_meter.avg)), dices_avg.append(float(dice_meter.avg))
                epochs.append(int(epoch)), bces.append(float(bce)), accs.append(float(acc)), dices.append(float(dice))
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(epochs, bces, 'g')
    axes[0].plot(epochs, bces_avg, 'g--')
    axes[0].set_title('bce')
    axes[1].plot(epochs, accs, 'r')
    axes[1].plot(epochs, accs_avg, 'r--')
    axes[1].set_title('acc')
    axes[2].plot(epochs, dices, 'b')
    axes[2].plot(epochs, dices_avg, 'b--')
    axes[2].set_title('dice')
    plt.savefig(args.loss_log.parent/'val_plots.png')
    with open(args.loss_log.parent/'val_results.json', 'w') as results_file:
        json.dump({
            'epochs': epochs, 'bce': bces, 'acc': accs, 'dice': dices,
            'avg_bce': bces_avg, 'avg_acc': accs_avg, 'avg_dice': dices_avg
        }, results_file)
    print("Done!")


