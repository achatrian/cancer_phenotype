from pathlib import Path
from argparse import ArgumentParser
import re
import json
from math import inf
import shutil
from base.utils.utils import AverageMeter


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('experiment_dir', type=Path)
    args = parser.parse_args()
    try:
        with open(args.experiment_dir/'val_results.json', 'r') as results_file:
            results = json.load(results_file)
        epochs = results['epochs']
    except (FileNotFoundError, KeyError):
        epochs, bces, accs, dices = [], [], [], []
        bces_avg, accs_avg, dices_avg = [], [], []
        bce_meter, acc_meter, dice_meter = AverageMeter(), AverageMeter(), AverageMeter()
        with (args.experiment_dir/'loss_log.txt').open('r') as loss_log_file:
            for line in loss_log_file:
                if 'validation' in line:
                    epoch, bce, acc, dice = re.search(
                        r'\(epoch: (\d*), validation\) bce: (\d*.?\d*) acc: (\d*.?\d*) dice: (\d*.?\d*)',
                        line).groups()
                    bce_meter.update(float(bce))
                    acc_meter.update(float(acc))
                    dice_meter.update(float(dice))
                    bces_avg.append(float(bce_meter.avg)), accs_avg.append(float(acc_meter.avg)), dices_avg.append(
                        float(dice_meter.avg))
                    epochs.append(int(epoch)), bces.append(float(bce)), accs.append(float(acc)), dices.append(
                        float(dice))
        results = {
                'epochs': epochs, 'bce': bces, 'acc': accs, 'dice': dices,
                'avg_bce': bces_avg, 'avg_acc': accs_avg, 'avg_dice': dices_avg
            }
        with open(args.experiment_dir / 'val_results.json', 'w') as results_file:
            json.dump(results, results_file)
    # make lists of all models present in directory and pick model with highest accuracy
    model_tags = [path.name for path in args.experiment_dir.glob('*.pth')]
    net_name = re.search(r'\d*_(net_\w*)', next(args.experiment_dir.glob('*.pth')).name).groups()[0]
    models_epochs = [int(model_tag.split('_')[0]) for model_tag in model_tags if model_tag.split('_')[0].isnumeric()]
    epochs = results['epochs']
    best_bce, best_epoch = inf, 0
    for model_epoch in models_epochs:
        absolute_difference = lambda value: abs(model_epoch - value)
        closest_saved_epoch = min(epochs, key=absolute_difference)
        bce_at_epoch = results['avg_bce'][epochs.index(model_epoch)]
        if bce_at_epoch < inf:
            best_bce = bce_at_epoch
            best_epoch = closest_saved_epoch
    shutil.copy(args.experiment_dir/f'{best_epoch}_{net_name}.pth', args.experiment_dir/f'best_{net_name}.pth')
    print(f"Best model found at epoch {best_epoch} with bce {best_bce}")



