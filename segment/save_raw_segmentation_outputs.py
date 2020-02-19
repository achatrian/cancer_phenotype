from pathlib import Path
from imageio import imwrite
from tqdm import tqdm
from base.options.test_options import TestOptions
from base.models import create_model
from base.datasets import create_dataset, create_dataloader
from base.utils.utils import tensor2im


if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = create_dataset(opt)
    dataset.setup()
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset=dataset)
    N = 20  # how many examples to save
    save_dir = Path(r'/well/rittscher/users/achatrian/debug/raw_unet_output')
    save_dir.mkdir(exist_ok=True, parents=True)
    image_counter = 0
    for n, data in enumerate(tqdm(dataloader)):
        model.set_input(data)
        model.optimize_parameters()
        model.evaluate_parameters()
        losses = model.get_current_losses()
        metrics = model.get_current_metrics()
        visuals, visuals_paths = model.get_current_visuals(), model.get_visual_paths()
        input_, output = visuals['input'], visuals['output']
        for image, map_ in zip(input_, output):
            image = tensor2im(image)
            map_ = tensor2im(map_, segmap=True, num_classes=opt.num_class)
            imwrite(save_dir/f'image_{image_counter}.png', image)
            imwrite(save_dir/f'segmentation_{image_counter}.png', map_)
            image_counter += 1
        if n == N:
            break



