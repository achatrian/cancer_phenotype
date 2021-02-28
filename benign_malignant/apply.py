from pathlib import Path
from datetime import datetime
import socket
import json
from tqdm import tqdm
import imageio
import numpy as np
import pandas as pd
import torch.nn.functional as f
from base.options.apply_options import ApplyOptions
from base.models import create_model
from base.datasets import create_dataset, create_dataloader


if __name__ == '__main__':
    opt = ApplyOptions().parse()
    print(f"Starting at {str(datetime.now())}")
    print(f"Running on host: '{socket.gethostname()}'")
    # hard-code some parameters for test
    opt.no_visdom = True
    opt.sequential_samples = True
    suffix_check = lambda path: path.name.endswith('.svs') or path.name.endswith('.ndpi') or \
                                path.name.endswith('.png') or path.name.endswith('.jpg')
    assert opt.num_class == 2
    image_paths = list(path for path in Path(opt.data_dir).iterdir() if suffix_check(path))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.png'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.jpg'))
    dataset = create_dataset(opt)
    dataset.setup()
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    if opt.eval:
        model.eval()
    save_dir = Path(opt.data_dir)/'data'/'experiments'/'benign_malignant'
    misclassified_dir = save_dir/'misclassified'
    save_dir.mkdir(exist_ok=True, parents=True)
    misclassified_dir.mkdir(exist_ok=True, parents=True)
    results, index = [], []
    misclassified_count = 0
    misclassified_labels = []
    for i, data in enumerate(tqdm(dataloader)):
        model.set_input(data)
        model.test()
        outputs = model.output.detach().cpu().numpy()
        if model.target is not None:
            targets = model.target.detach().cpu().numpy()
        else:
            targets = [-1]*len(outputs)
        output_probabilities = f.softmax(model.output, dim=1).detach().cpu().numpy()
        for j, (probabilites, target) in enumerate(zip(output_probabilities, targets)):
            positive_probability, target = float(probabilites[1]), int(target)
            x, y, w, h = int(data['x_offset'][j]), int(data['y_offset'][j]), \
                         int(data['width'][j]), int(data['height'][j])
            box_id, slide_id = f'{x}_{y}_{w}_{h}', Path(data['input_path'][j]).parent.name
            results.append(dict(
                positive_probability=positive_probability, label=target,
                input_path=data['input_path'][j], box_id=box_id, slide_id=slide_id,
                x=x, y=y, w=w, h=h
            ))
            index.append(box_id)
            if int(positive_probability >= 0.5) != target and target != -1:
                misclassified_count += 1
                misclassified_labels.append(target)
                image = data['input'][j].detach().cpu().numpy().transpose(1, 2, 0)
                image = (image * 0.5 + 0.5) * 255.0
                imageio.imwrite(misclassified_dir/
                                f'{model.model_tag}_{Path(data["input_path"][j]).name}',
                                image.astype(np.uint8))
    results_df = pd.DataFrame(data=results, index=index,
                             columns=['positive_probability', 'label', 'input_path', 'box_id', 'slide_id', 'x', 'y', 'w', 'h'])
    results_path = save_dir/f'{opt.experiment_name}_{model.model_tag}.csv'
    with open(results_path, 'w') as results_file:
        results_df.to_csv(results_file)
    if model.target is not None:
        print(f"""{misclassified_count}/{len(dataset)} tiles were misclassified 
              -- {len(misclassified_labels) - sum(misclassified_labels)} for label 0
              and {sum(misclassified_labels)} for label 1""")
    print(f"Saved results to {results_path}")



