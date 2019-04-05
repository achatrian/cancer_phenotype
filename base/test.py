from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from options.test_options import TestOptions
from data import create_dataset, create_dataloader
from models import create_model
from utils.base_visualizer import save_images
from utils import html, create_visualizer, utils

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
    dataset.setup()
    if opt.make_subset:
        dataset.make_subset()
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    # create a website
    web_dir = Path(opt.data_dir)/opt.experiment_name/f'{opt.phase}_{opt.load_epoch}'
    webpage = html.HTML(str(web_dir), f'Experiment = {opt.experiment_name}, Phase = {opt.phase}, Epoch = {opt.load_epoch}')
    if opt.eval:
        model.eval()
    print("Begin testing ...")
    with model.start_validation() as update_validation_meters:
        with tqdm(total=len(dataset)) as progress_bar:
            for i, data in enumerate(dataloader):
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
    utils.mkdirs(save_results_dir)
    results_name = model.model_tag + f'_{opt.slide_id}' if opt.slide_id else ''
    with open((save_results_dir/results_name).with_suffix('.json'), 'a+') as results_json:
        try:
            results = json.load(results_json)
        except json.decoder.JSONDecodeError:
            results = dict()
        if opt.slide_id and opt.slide_id in results:
            slide_results = results[opt.slide_id]
        else:
            slide_results = dict()
        slide_results['date'] = str(datetime.now())
        slide_results['model'] = model.model_tag
        slide_results['split'] = Path(opt.split_file).with_suffix('').name
        slide_results['options'] = TestOptions().print_options(opt, True).split('\n')
        slide_results['epoch'] = opt.load_epoch
        slide_results['message'] = message
        slide_results['losses'] = losses
        slide_results['metrics'] = metrics
        if opt.slide_id:
            results[opt.slide_id] = slide_results
        else:
            results[model.model_tag] = slide_results
        json.dump(results, results_json)
    print("Done!")
