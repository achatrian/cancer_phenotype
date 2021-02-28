from pathlib import Path
from datetime import datetime
from collections import deque
from numbers import Integral
import json
import math
import numpy as np
import cv2
from matplotlib import cm
import torch
import torch.nn.functional as f
from tqdm import tqdm
from torchvision.utils import make_grid
import imageio
from captum.attr import DeepLift, visualization
from utils.guided_backprop import GuidedBackprop
from options.attribution_options import AttributionOptions
from datasets import create_dataset, create_dataloader
from models import create_model
from utils import utils, debug


r"""Code for creating salience maps from guided backpropagation gradients on neural network.
Limitation: need nn.Model instance that has .features attribute"""

jet = cm.get_cmap('jet')


def convert_to_jet(image):
    r""" Visualize deeplift (unlike for integrated gradients, the absolute value is not taken)
    :param image:
    :return:
    """
    normalized = ((image - image.min())/(image.max() - image.min()))  # make everything positive
    spatial = normalized.sum(axis=2)
    spatial_normalized = ((spatial - np.percentile(spatial, 1))/(np.percentile(spatial, 99) - np.percentile(spatial, 1)))
    spatial_normalized = np.clip(spatial_normalized.astype(np.float), 0.0, 1.0)
    coloured = jet(spatial_normalized)
    return coloured


def overlay_grids(example_grid, attribution, threshold=0.1, weights=(0.8, 0.2)):
    r"""overlay images and gradient grids by alpha blending"""
    # https://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html
    # overlay two color grids, one of images and one of saliency maps
    attribution[attribution < threshold] = 0.0
    attribution = attribution * 255.0  # gradient grid goes from 0 to 1
    example_grid = (example_grid + 1) / 2.0 * 255.0  # example grid goes from -1 to 1
    return cv2.addWeighted(example_grid.astype(np.uint8), weights[0], attribution.astype(np.uint8), weights[1], 0)


if __name__ == '__main__':
    opt = AttributionOptions().parse()
    opt.display_id = -1   # no visdom display
    opt.sequential_samples = True
    dataset = create_dataset(opt)
    dataset.setup()
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    confidence_struct = dict()  # slide_id -> ( example_id (location) -> (confidence probability, target_class))
    to_attribute = dict()
    module = getattr(model, opt.target_network)
    # Walk module hierarchy till module with submodule named 'features' is found
    # do this here to check that module to back-prop exists before processing all the data (fail fast)
    if opt.target_module is not None:
        for key, submodule in module.named_modules():
            module_hierarchy = key.split('.')
            if module_hierarchy[-1] == opt.target_module:
                net = module
                for module_name in module_hierarchy[:-1]:
                    net = getattr(net, module_name)
                break
        else:
            raise ValueError(f"Module {opt.target_network} has no attribute '{opt.target_module}'")
    else:
        net = module
    # calculate how many entries should be in top_examples in order to obtain the desired number of salience maps
    labelled_slides = [set() for i in set(dataset.labels)]
    for i, (path, label) in enumerate(zip(dataset.paths, dataset.labels)):
        slide_id = path.parent.name
        labelled_slides[label].add(slide_id)
    num_examples_per_label = [opt.n_salience_maps*opt.n_grid_examples//len(slides) for slides in labelled_slides]
    # Get confidence probability for all dataset items (slides), and store subset of high-confidence examples
    print("Finding tiles with most confident prediction ...")
    for data in tqdm(dataloader):
        model.set_input(data)
        model.test()
        model.evaluate_parameters()
        output_probs = f.softmax(model.output, dim=1).detach().cpu().numpy()
        for i, confidence_probs in enumerate(output_probs):
            slide_id = Path(data['input_path'][i]).parent.name  # assuming dir containing tiles is named after the slide
            input_path = data['input_path'][i]
            target = data['target'][i].cpu().numpy().item()  # assume target is number,
            if slide_id in confidence_struct:
                examples_confidence = confidence_struct[slide_id]
                top_examples = to_attribute[slide_id]
            else:
                confidence_struct[slide_id] = examples_confidence = dict()  # add slide to index of probability data
                to_attribute[slide_id] = top_examples = deque(maxlen=num_examples_per_label[target])
            class_prob = confidence_probs[target]
            if not isinstance(target, Integral) and not 0 < target <= opt.num_clas:
                raise ValueError(f"Targets must be positive integers (type: {type(target)})")
            # images id is its coordinate tuple relative to slide origin (hence it is hashable)
            example_id = (data['x_offset'][i].cpu().numpy().item(), data['y_offset'][i].cpu().numpy().item())
            if 'tile_num' in data:
                example_id = example_id + (data['tile_num'][i].cpu().numpy().item(),)
            examples_confidence[example_id] = (confidence_probs, target)
            example = data['input'][i]
            for j, (id_, prob, t, ex, path) in enumerate(top_examples):
                if class_prob > prob and t == target:
                    if len(top_examples) == top_examples.maxlen:
                        del top_examples[j]  # calling insert() on full deque raises error
                    top_examples.insert(j, (example_id, class_prob, target, example, input_path))
                    break
            else:  # executes if for loop does not break
                if len(top_examples) < top_examples.maxlen:
                    top_examples.appendleft((example_id, class_prob, target, example, input_path))  # keeps increasing order
    # Perform guided backprop on top confidence examples
    save_dir = Path(opt.checkpoints_dir)/opt.experiment_name/'attributions'
    if hasattr(opt, 'split_file'):
        save_dir = save_dir/Path(opt.split_file).with_suffix('').name
    save_dir.mkdir(exist_ok=True, parents=True)
    if opt.method == 'deeplift':
        deep_lift = DeepLift(net)

        def attribute(example, target):
            example = example.unsqueeze(0).requires_grad_(True)
            attribution = deep_lift.attribute(example, target=target)
            attribution = attribution.detach().cpu().numpy()  #
            attribution = convert_to_jet(attribution.squeeze())  # change this to see +ve and -ve contributions
            # FIXME reducing over wrong axis
            return attribution
    elif opt.method == 'guidedbp':
        guided_backprop = GuidedBackprop(net, first_layer_name=opt.target_module)

        def attribute(example, target):
            # Get gradients
            guided_grads = guided_backprop.generate_gradients(example, target, is_cuda=bool(opt.gpu_ids))
            # FIXME not returning grads at image
            # Save colored gradients
            # utils.save_gradient_images(guided_grads, save_dir/f'{example_id[0]}_{example_id[1]}_Guided_BP_color.png')
            # Convert to grayscale
            grayscale_guided_grads = convert_to_jet(np.transpose(guided_grads, (1, 2, 0)))  # TODO test transpose
            # Save grayscale gradients
            # utils.save_gradient_images(grayscale_guided_grads, save_results_dir/f'{example_id}_Guided_BP_gray')
            # # Positive and negative saliency maps
            # pos_sal, neg_sal = utils.get_positive_negative_saliency(guided_grads)
            # utils.save_gradient_images(pos_sal, save_results_dir/f'{example_id}_pos_sal')
            # utils.save_gradient_images(neg_sal, save_results_dir/f'{example_id}_neg_sal')
            return grayscale_guided_grads
    else:
        raise ValueError(f"Unrecognized attribution method '{opt.method}'")
    counter = 0
    print(f"Start backpropagating on {sum(len(to_attribute[k]) for k in to_attribute)} tiles from {len(to_attribute)} slides ...")
    examples_for_grid, attributions_for_grid = list([] for prob in confidence_probs), list([] for prob in confidence_probs)
    grids_info = list([[]] for prob in confidence_probs)  # contains information on each tile in first grid for each class
    with tqdm(total=sum(len(examples) for examples in to_attribute.values())) as pbar:
        for slide_id in to_attribute:
            for example_id, class_prob, target, example, input_path in to_attribute[slide_id]:
                attribution = attribute(example, target)
                attribution = torch.from_numpy(attribution.transpose(2, 0, 1))
                examples_for_grid[target].append(example)
                attributions_for_grid[target].append(attribution)
                grids_info[target][-1].append({
                    'tile_num': len(grids_info[target][-1]),
                    'slide': slide_id,
                    'class': int(target),
                    'probability': float(class_prob),
                    'coords': example_id})
                if len(examples_for_grid[target]) == opt.n_grid_examples:
                    example_grid = make_grid(examples_for_grid[target], nrow=round(math.sqrt(opt.n_grid_examples)))  # get grids
                    attribution_grid = make_grid(attributions_for_grid[target], nrow=round(math.sqrt(opt.n_grid_examples)))
                    examples_for_grid[target].clear()
                    attributions_for_grid[target].clear()
                    example_grid = example_grid.cpu().numpy().transpose(1, 2, 0)
                    attribution_grid = attribution_grid.cpu().numpy().transpose(1, 2, 0)
                    imageio.imwrite(save_dir/f'examples_class{target}_{len(grids_info[target])}.png', example_grid)
                    imageio.imwrite(save_dir /f'attribution_class{target}_{len(grids_info[target])}.png', attribution_grid)
                    imageio.imwrite(save_dir /f'overlaid_class{target}_{len(grids_info[target])}.png',
                                    overlay_grids(example_grid, attribution_grid))
                    grids_info[target].append([])
                    counter += 1
                pbar.update()  # FIXME loop ends before bar is completed
    json.dump(grids_info, open(save_dir/f'grids_info_{opt.method}.json', 'w'))
    print(f'Attribution completed - {counter} images were saved to {save_dir}')

