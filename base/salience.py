from pathlib import Path
from collections import deque
from numbers import Integral
import torch.nn.functional as f
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
    # Get confidence probability for all dataset subdivisions (slides)
    for data in dataloader:
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
                confidence_struct[slide_id] = examples_confidence = dict()
                to_backprop[slide_id] = top_examples = deque(maxlen=opt.n_salience_maps)
            target = data['target'][i].cpu().numpy()  # assume target is number,
            class_prob = confidence_probs[target]
            if not isinstance(target, Integral) or target < 0:
                raise ValueError(f"Targets must be positive integers (type: {type(target)})")
            example_id = (data['x_offset'], data['y_offset'])
            examples_confidence[example_id] = (confidence_probs, target)
            example = data['input'][i]
            for j, (id_, prob, t, ex) in top_examples:
                if class_prob > prob:
                    top_examples.insert(j, (example_id, class_prob, target, example))
                    break
            else:  # executes if for loop does not break
                if len(top_examples) < top_examples.maxlen:
                    top_examples.appendleft((example_id, class_prob, target, example))  # keeps increasing order
    # Guided backprop on top confidence examples
    save_results_dir = Path(opt.checkpoints_dir)/opt.experiment_name/'results'/f'guided_backprop_{model.model_tag}'
    net = getattr(model, opt.target_network)
    guided_backprop = GuidedBackprop(net)
    for slide_id in to_backprop:
        for example_id, class_prob, target, example in to_backprop[slide_id]:
            # Get gradients
            guided_grads = guided_backprop.generate_gradients(example, target)
            # Save colored gradients
            utils.save_gradient_images(guided_grads, save_results_dir/f'{example_id}_Guided_BP_color')
            # Convert to grayscale
            # grayscale_guided_grads = utils.convert_to_grayscale(guided_grads)
            # Save grayscale gradients
            # utils.save_gradient_images(grayscale_guided_grads, save_results_dir/f'{example_id}_Guided_BP_gray')
            # # Positive and negative saliency maps
            # pos_sal, neg_sal = utils.get_positive_negative_saliency(guided_grads)
            # utils.save_gradient_images(pos_sal, save_results_dir/f'{example_id}_pos_sal')
            # utils.save_gradient_images(neg_sal, save_results_dir/f'{example_id}_neg_sal')
    print('Guided backprop completed')


