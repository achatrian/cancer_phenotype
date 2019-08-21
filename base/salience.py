from pathlib import Path
from collections import deque
from numbers import Integral
import json
import math
import torch
import torch.nn.functional as f
from tqdm import tqdm
from torchvision.utils import make_grid
import imageio
from utils.guided_backprop import GuidedBackprop
from options.salience_options import SalienceOptions
from data import create_dataset, create_dataloader
from models import create_model
from utils import utils

r"""Code for creating salience maps from guided backpropagation gradients on neural network.
Limitation: need nn.Model instance that has .features attribute"""

if __name__ == '__main__':
    opt = SalienceOptions().parse()
    opt.display_id = -1   # no visdom display
    dataset = create_dataset(opt)
    dataset.setup()
    if opt.make_subset:
        dataset.make_subset()
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    confidence_struct = dict()  # slide_id -> ( example_id (location) -> (confidence probability, target_class))
    to_backprop = dict()
    # Get confidence probability for all dataset subdivisions (slides), and store subset of high-confidence examples
    print("Finding tiles with most confident prediction ...")
    for data in tqdm(dataloader):
        model.set_input(data)
        model.test()
        model.evaluate_parameters()
        output_probs = f.softmax(model.output, dim=1).detach().cpu().numpy()
        for i, confidence_probs in enumerate(output_probs):
            slide_id = Path(data['input_path'][i]).parent.name
            if slide_id in confidence_struct:
                examples_confidence = confidence_struct[slide_id]
                top_examples = to_backprop[slide_id]
            else:
                confidence_struct[slide_id] = examples_confidence = dict()  # add slide to index of probability data
                to_backprop[slide_id] = top_examples = deque(maxlen=opt.n_salience_maps)
            target = data['target'][i].cpu().numpy().item()  # assume target is number,
            class_prob = confidence_probs[target]
            if not isinstance(target, Integral) or target < 0:
                raise ValueError(f"Targets must be positive integers (type: {type(target)})")
            # images id is its coordinate tuple relative to slide origin (hence it is hashable)
            example_id = (data['x_offset'][i].cpu().numpy().item(), data['y_offset'][i].cpu().numpy().item())
            examples_confidence[example_id] = (confidence_probs, target)
            example = data['input'][i]
            for j, (id_, prob, t, ex) in enumerate(top_examples):
                if class_prob > prob:
                    if len(top_examples) == top_examples.maxlen:
                        del top_examples[j]  # calling insert() on full deque raises error
                    top_examples.insert(j, (example_id, class_prob, target, example))
                    break
            else:  # executes if for loop does not break
                if len(top_examples) < top_examples.maxlen:
                    top_examples.appendleft((example_id, class_prob, target, example))  # keeps increasing order
    module = getattr(model, opt.target_network)
    # Walk module hierarchy till module with submodule named 'features' is found
    for key, submodule in module.named_modules():
        module_hierarchy = key.split('.')
        if module_hierarchy[-1] == opt.target_module:
            net = module
            for module_name in module_hierarchy[:-1]:
                net = getattr(net, module_name)
            break
    else:
        raise ValueError(f"Module {opt.target_network} has no attribute 'features'")
    # Perform guided backprop on top confidence examples
    save_dir = Path(opt.checkpoints_dir)/opt.experiment_name/'saliency_maps'/Path(opt.split_file).with_suffix('').name
    guided_backprop = GuidedBackprop(net)
    counter = 0
    print("Start backpropagating ...")
    examples_for_grid, grads_for_grid = list([] for prob in confidence_probs), list([] for prob in confidence_probs)
    grids_info = list([[]] for prob in confidence_probs)  # contains information on each tile in first grid for each class
    with tqdm(total=opt.n_salience_maps*len(to_backprop)) as pbar:
        for slide_id in to_backprop:
            for example_id, class_prob, target, example in to_backprop[slide_id]:
                # Get gradients
                guided_grads = guided_backprop.generate_gradients(example, target, is_cuda=bool(opt.gpu_ids))
                # Save colored gradients
                #utils.save_gradient_images(guided_grads, save_dir/f'{example_id[0]}_{example_id[1]}_Guided_BP_color.png')
                # Convert to grayscale
                grayscale_guided_grads = utils.convert_to_grayscale(guided_grads)
                # Save grayscale gradients
                # utils.save_gradient_images(grayscale_guided_grads, save_results_dir/f'{example_id}_Guided_BP_gray')
                # # Positive and negative saliency maps
                # pos_sal, neg_sal = utils.get_positive_negative_saliency(guided_grads)
                # utils.save_gradient_images(pos_sal, save_results_dir/f'{example_id}_pos_sal')
                # utils.save_gradient_images(neg_sal, save_results_dir/f'{example_id}_neg_sal')
                examples_for_grid[target].append(example)
                grads_for_grid[target].append(torch.from_numpy(grayscale_guided_grads))
                grids_info[target][-1].append({
                    'tile_num': len(grids_info[target][-1]),
                    'slide': slide_id,
                    'class': int(target),
                    'probability': float(class_prob),
                    'coords': example_id})
                if len(examples_for_grid[target]) == opt.n_grid_examples:
                    example_grid = make_grid(examples_for_grid[target], nrow=round(math.sqrt(opt.n_grid_examples)))  # get grids
                    grad_grid = make_grid(grads_for_grid[target], nrow=round(math.sqrt(opt.n_grid_examples)))
                    examples_for_grid[target].clear()
                    grads_for_grid[target].clear()
                    example_grid = example_grid.cpu().numpy().transpose(1, 2, 0)
                    grad_grid = grad_grid.cpu().numpy().transpose(1, 2, 0)
                    imageio.imwrite(save_dir/f'examples_class{target}_{len(grids_info[target])}.png', example_grid)
                    imageio.imwrite(save_dir/f'grads_class{target}_{len(grids_info[target])}.png', grad_grid)
                    imageio.imwrite(save_dir/f'overlaid_class{target}_{len(grids_info[target])}.png',
                                    utils.overlay_grids(example_grid, grad_grid))
                    grids_info[target].append([])
                    counter += 1
                pbar.update()
    json.dump(grids_info, open(save_dir/'grids_info.json', 'w'))
    print(f'Guided backprop completed - {counter} images were saved')


