from pathlib import Path
import json
from options.test_options import TestOptions
from data import create_dataset, create_dataloader
from models import create_model
from utils.base_visualizer import save_images
from utils import html, create_visualizer

r"Test script for network, aggregates results over whole validation dataset"

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    dataset = create_dataset(opt)
    if opt.slide_id:  # remove all paths not belonging to desired slide -- id must be contained in path to file
        indices = [i for i, path in enumerate(dataset.paths) if opt.slide_id in str(path)]  # NB only works for datasets that store paths in self.paths
        dataset = dataset.make_subset(indices)
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup()
    # create a website
    web_dir = Path(opt.data_dir)/opt.experiment_name/f'{opt.phase}_{opt.load_epoch}'
    webpage = html.HTML(str(web_dir), f'Experiment = {opt.experiment_name}, Phase = {opt.phase}, Epoch = {opt.load_epoch}')
    if opt.eval:
        model.eval()
    with model.start_validation() as update_validation_meters:
        for i, data in enumerate(dataloader):
            model.set_input(data)
            model.test()
            model.evaluate_parameters()
            update_validation_meters()
            visuals = model.get_current_visuals()
            visual_paths = model.get_visual_paths()
        save_images(webpage, visuals, visual_paths['input'], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # save the website
    webpage.save()
    # write results
    message = create_visualizer(opt).print_current_losses_metrics(
        opt.load_epoch,
        None,
        model.get_current_losses(),
        model.get_current_metrics())
    with open(Path(opt.checkpoints_dir)/'results'/model.model_tag, 'a+') as results_json:
        try:
            results = json.load(results_json)
        except FileNotFoundError:
            results = dict()
        if opt.slide_id and opt.slide_id in results:
            slide_results = results[opt.slide_id]
        else:
            slide_results = dict()
        slide_results['model'] = model.model_tag
        slide_results['split'] = opt.split_file
        slide_results['options'] = TestOptions().print_options(opt)
        slide_results['epoch'] = opt.load_epoch
        slide_results['message'] = message
        slide_results['losses'] = dict(model.get_current_losses())
        slide_results['metrics'] = dict(model.get_current_metrics())
        if opt.slide_id:
            results[opt.slide_id] = slide_results
        else:
            results[model.model_tag] = slide_results
    print("Done!")
