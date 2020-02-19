import socket
from pathlib import Path
import json
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from base.options.test_options import TestOptions
from base.datasets import create_dataset, create_dataloader
from base.models import create_model
r"Test script for network, aggregates results over whole validation dataset"


if __name__ == '__main__':
    opt = TestOptions().parse()
    print(f"Running on host: '{socket.gethostname()}'")
    # hard-code some parameters for test
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    dataset = create_dataset(opt)
    # dataloader uses a reference to dataset, and random sampler recomputes the number of samples each time, so changing the dataset to the dataloader should work
    # https://discuss.pytorch.org/t/solved-will-change-in-dataset-be-reflected-on-dataloader-automatically/10206/2
    model = create_model(opt)
    model.setup(dataset)
    if opt.eval:
        model.eval()
    print("Begin testing ...")
    tiles_dir = Path(opt.data_dir)/'data'/'tiles'
    foci_paths = tuple(tiles_dir.iterdir())
    all_results = {}
    foci_results = all_results['results'] = []
    test_errors = all_results['test_errors'] = []
    all_targets, all_outputs = [], []
    dataset = create_dataset(opt, print_dataset_info=False)
    for focus_path in tqdm(foci_paths, desc='focus'):
        if not focus_path.is_dir():
            continue
        slides_paths = tuple(focus_path.iterdir())
        for slide_path in tqdm(slides_paths, desc='slide'):
            # only select test images in this subfolder
            try:
                focus_dataset = dataset.make_subset(selector=str(slide_path), deepcopy=True)  # TODO TEST if changing reference works
            except ValueError as err:
                test_errors.append({'error': str(err), 'slide': str(slide_path)})
                print(f"There are no available tiles for focus image: '{str(slide_path)}'")
                continue
            focus_dataset.setup()
            dataloader = create_dataloader(focus_dataset)
            for meter in model.meters.values():
                meter.reset()  # reset meters, so that results are aggregated for one focus at a time
            # save positive probabilities and targets in order to compute AUC
            targets, class_predictions = [], []
            for data in dataloader:
                model.set_input(data)
                model.test()
                model.evaluate_parameters()
                target = model.target.detach().cpu().numpy().squeeze()
                output = torch.nn.functional.softmax(model.output, dim=1).max(1)[1].detach().cpu().numpy().astype(target.dtype).squeeze().tolist()
                target = target.tolist()  # convert later as need to extract dtype for above
                try:
                    targets.extend(target)
                except TypeError:
                    targets.append(target)
                try:
                    class_predictions.extend(output)
                except TypeError:
                    class_predictions.append(output)
            # Compute the probability that focus belongs to positive class by averaging over how many tiles which compose it
            # belong to that class. In case there is only one tile, its probability becomes the probability for the whole focus
            focus_class_prob = np.mean(class_predictions)
            losses, metrics = model.get_current_metrics(), model.get_current_losses()
            focus_results = dict(
                focus=focus_path.name, slide_id=slide_path.name, target=int(targets[0]), num_tiles=len(dataloader),
                class_prob=focus_class_prob, prediction_acc=float(np.mean(class_predictions == targets)),
                                 )
            focus_results.update(**dict(losses, **metrics))  # merge
            print(f"Results for {slide_path.name}, {focus_path.name}:")
            print(' '.join([f'{k}={v},' for k, v in focus_results.items()]))
            foci_results.append(focus_results)
    # compute roc auc score for all foci together
    all_results['auc'] = roc_auc_score(
        [focus_result['target'] for focus_result in foci_results],
        [focus_result['class_prob'] for focus_result in foci_results]
    )
    all_results['accuracy'] = float(np.mean([focus_result['prediction_acc'] for focus_result in foci_results]))
    with open(Path(opt.checkpoints_dir)/opt.experiment_name/f'foci_test_results_e:{opt.load_epoch}.json', 'w') as foci_results_file:
        json.dump(all_results, foci_results_file)
    print("Done !")





