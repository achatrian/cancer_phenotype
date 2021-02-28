import socket
from pathlib import Path
import json
from numbers import Real
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from base.options.test_options import TestOptions
from base.datasets import create_dataset, create_dataloader
from base.models import create_model
from ihc.datasets.ihcpatch_dataset import IHCPatchDataset
r"Test script for network, aggregates results over whole validation dataset"


if __name__ == '__main__':
    opt = TestOptions().parse()  # only select test foci
    print(f"Running on host: '{socket.gethostname()}'")
    # hard-code some parameters for test
    opt.sequential_samples = True  # no shuffle or weighted sampling
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.overwrite_split = False
    dataset = create_dataset(opt)
    # dataloader uses a reference to dataset, and random sampler recomputes the number of samples each time, so changing the dataset to the dataloader should work
    # https://discuss.pytorch.org/t/solved-will-change-in-dataset-be-reflected-on-dataloader-automatically/10206/2
    model = create_model(opt)
    model.setup(dataset)
    if opt.eval:
        model.eval()
    print("Begin testing ...")
    tiles_dir = Path(opt.data_dir)/'data'/opt.tiles_dirname
    foci_paths = tuple(tiles_dir.iterdir())
    all_results = {}
    foci_results = all_results['results'] = []
    test_errors = all_results['test_errors'] = []
    all_targets, all_outputs = [], []
    dataset: IHCPatchDataset = create_dataset(opt, print_dataset_info=False)
    slides_data = dataset.slides_data
    for focus_path in tqdm(foci_paths, desc='focus'):
        if not focus_path.is_dir():
            continue
        slides_paths = tuple(focus_path.iterdir())
        for slide_path in tqdm(slides_paths, desc='slide'):
            # only select test images in this sub-folder
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
            targets, class_predictions, variances, loss_variances = [], [], [], []
            for data in dataloader:  # TODO remove need for creating dataloader, as it consumes a lot of time
                model.set_input(data)
                model.test()
                model.evaluate_parameters()
                target = model.target.detach().cpu().numpy().squeeze().tolist()
                output = torch.nn.functional.softmax(model.output, dim=1)[:, 1].detach().cpu().numpy().astype(float).squeeze().tolist()
                try:
                    targets.extend(target)
                    class_predictions.extend(output)
                except TypeError:
                    targets.append(target)
                    class_predictions.append(output)
                if hasattr(model, 'variance'):  # TODO test attributes of ensemble model
                    variance = model.variance.detach().cpu().numpy().squeeze().sum().tolist()  # reported variance is the sum of variances over classes
                    loss_variance = model.loss_variance.detach().cpu().numpy().squeeze().tolist()
                    try:
                        variances.extend(variance)
                        loss_variances.extend(loss_variance)
                    except TypeError:
                        variances.append(variance)
                        loss_variances.append(loss_variance)
            # Compute the probability that focus belongs to positive class by averaging over how many tiles which compose it
            # belong to that class. In case there is only one tile, its probability becomes the probability for the whole focus
            focus_class_prob = np.mean(class_predictions)
            losses, metrics = model.get_current_metrics(), model.get_current_losses()
            focus_results = dict(
                focus=focus_path.name, slide_id=slide_path.with_suffix('').name, target=int(targets[0]), num_tiles=len(dataloader),
                class_prob=focus_class_prob, prediction_acc=float(np.mean(np.round(class_predictions) == targets)),
                                 )
            focus_results.update(**dict(losses, **metrics))  # merge
            if hasattr(model, 'variance'):  # TODO test attributes of ensemble model
                focus_results.update(loss_variance=float(np.mean(loss_variances)), variance=float(sum(variances)))
            print(f"Results for {slide_path.name}, {focus_path.name}:")
            print(' '.join([f'{k}={v},' for k, v in focus_results.items()]))
            foci_results.append(focus_results)
    # compute roc auc score for all foci together
    all_results['auc'] = roc_auc_score(
        [focus_result['target'] for focus_result in foci_results],
        [focus_result['class_prob'] for focus_result in foci_results]
    )
    all_results['accuracy'] = float(np.mean([focus_result['acc'] for focus_result in foci_results]))
    # break down results by ihc reason + no ihc
    results_by_ihc_reason = {}
    try:
        for focus_result in foci_results:
            slide_data = slides_data.loc[
                (slides_data['Image'] == focus_result['slide_id']) &
                (slides_data['Focus number'] == int(focus_result['focus'][-1]))
                ]
            if slide_data.empty:
                continue
            if len(slide_data) > 1:
                slide_data = slide_data.iloc[0]
            if isinstance(slide_data['IHC reason'], float):
                ihc_reason = -1
            elif isinstance(slide_data['IHC reason'], str):
                ihc_reason = slide_data['IHC reason'][0]
            elif isinstance(slide_data['IHC reason'].item(), str):
                ihc_reason = int(slide_data['IHC reason'].item()[0])
            else:
                ihc_reason = -1
            if ihc_reason not in results_by_ihc_reason:
                results_by_ihc_reason[ihc_reason] = {
                    name: [] for name, value in focus_result.items() if isinstance(value, Real)
                }
            for name, value in focus_result.items():
                if isinstance(value, Real):
                    results_by_ihc_reason[ihc_reason][name].append(value)
        for ihc_reason in results_by_ihc_reason:
            for name in results_by_ihc_reason[ihc_reason]:
                results_by_ihc_reason[ihc_reason][name] = np.mean(results_by_ihc_reason[ihc_reason][name])
    except Exception as err:
        print("Couldn't assign ihc reasons")
        print(err)
    all_results['ihc_reason_breakdown'] = results_by_ihc_reason
    from datetime import datetime
    (Path(opt.checkpoints_dir)/opt.experiment_name).mkdir(exist_ok=True, parents=True)
    with open(Path(opt.checkpoints_dir)/opt.experiment_name/f'foci_test_results_e:{opt.load_epoch}_{str(datetime.now())}.json', 'w') as foci_results_file:
        json.dump(all_results, foci_results_file)
    print("Done !")

