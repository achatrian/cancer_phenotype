from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from options.test_options import TestOptions
from data import create_dataset, create_dataloader
from models import create_model
from utils.base_visualizer import save_images
from utils import html_, create_visualizer, utils
r"Test script for network, aggregates results over whole validation dataset"

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    if opt.slide_id:
        opt.slide_id = str(Path(opt.slide_id).with_suffix(''))
    dataset = create_dataset(opt)
    if opt.make_subset:
        dataset.make_subset()
    dataset.setup()  # NB swapped in position .make_subset() and .setup()
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    # create a website
    web_dir = Path(opt.data_dir)/opt.experiment_name/f'{opt.phase}_{opt.load_epoch}'
    webpage = html_.HTML(str(web_dir), f'Experiment = {opt.experiment_name}, Phase = {opt.phase}, Epoch = {opt.load_epoch}')
    if opt.eval:
        model.eval()
    print("Begin testing ...")
    slide_level_label = set()
    with model.start_validation() as update_validation_meters:
        with tqdm(total=len(dataset)) as progress_bar:
            for i, data in enumerate(dataloader):
                slide_level_label.add(int(data['target'][0]))
                if len(slide_level_label) > 1:
                    raise ValueError(f"Inconsistent slide-level label for {opt.slide_id}; labels = {slide_level_label}")
                model.set_input(data)
                model.test()
                model.evaluate_parameters()
                update_validation_meters()
                visuals = model.get_current_visuals()
                visual_paths = model.get_visual_paths()
                progress_bar.update(n=model.input.shape[0])
        save_images(webpage, visuals, visual_paths['input'], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        losses, metrics = model.get_current_losses(), model.get_current_metrics()  # test measures
    # save the website
    webpage.save()
    # write results
    message = create_visualizer(opt).print_current_losses_metrics(opt.load_epoch, None, losses, metrics)
    save_results_dir = Path(opt.checkpoints_dir)/opt.experiment_name/'results'/Path(opt.split_file).with_suffix('').name
    utils.mkdirs(save_results_dir/model.model_tag)
    split_name = Path(opt.split_file).with_suffix('').name
    results_name = f'{opt.slide_id}' if opt.slide_id else split_name
    # save results for individual slide
    results = {
        'id': opt.slide_id if opt.slide_id else Path(opt.split_file).with_suffix('').name,
        'date': str(datetime.now()),
        'model': model.model_tag,
        'split': split_name,
        'options': TestOptions().print_options(opt, True).split('\n'),
        'epoch': opt.load_epoch,
        'message': message,
        'losses': losses,
        'metrics': metrics,
        'slide_level_label': slide_level_label.pop()
    }
    json.dump(results, open((save_results_dir/model.model_tag/results_name).with_suffix('.json'), 'w'))
    # gather results for all slides
    model_results = dict(options=results['options'])
    model_results['slides'] = slides_results = dict()
    counter = 0
    data_losses = {name: [] for name in losses}
    data_metrics = {name: [] for name in metrics}
    for result_path in (save_results_dir/model.model_tag).iterdir():
        with open(result_path, 'r') as result_json:
            slide_results = json.load(result_json)
            # store the slide level results for final averaging
            for name, value in slide_results['losses'].items():
                data_losses[name].append(value)
            for name, value in slide_results['metrics'].items():
                data_metrics[name].append(value)
            slides_results[slide_results['id']] = slide_results
            del slides_results[slide_results['id']]['options']
            counter += 1
    model_results['num_cases'] = counter
    model_results['statistics'] = {
        'losses': {name: {
                    'mean': np.mean(data_losses[name]),
                    'std': np.std(data_losses[name])
                    } for name in data_losses},
        'metrics': {name: {
                    'mean': np.mean(data_metrics[name]),
                    'std': np.std(data_metrics[name])
                    } for name in data_metrics}
    }
    # compute ROC AUC score over all slides
    results = sorted(slides_results.values(), key=lambda result: result['id'])  # ensure order is same for arrays below
    slide_level_labels = np.fromiter((result['slide_level_label'] for result in results), float)
    slide_pos_probs = np.fromiter((result['metrics']['pos_prob_val'] for result in results), float)
    model_results['statistics']['roc_auc_score'] = roc_auc_score(slide_level_labels, slide_pos_probs)
    json.dump(model_results, open((save_results_dir / model.model_tag).with_suffix('.json'), 'w'))
    print(f"Done! Model results for {model.model_tag} on {model_results['num_cases']} slides are available")
