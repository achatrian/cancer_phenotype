import os
from options.test_options import TestOptions
from data import create_dataloader
from models import create_model
from utils.base_visualizer import save_images
from utils.base_visualizer import Visualizer
from utils import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    dataloader = create_dataloader(opt)
    dataset = dataloader.load_data()
    model = create_model(opt)
    model.setup()
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
    with model.start_validation() as update_validation_meters:
        for i, data in enumerate(dataset):
            for j, data in enumerate(dataloader):
                model.set_input(data)
                model.test()
                model.evaluate_parameters()
                update_validation_meters()
                visuals = model.get_current_visuals()
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # save the website
    webpage.save()
