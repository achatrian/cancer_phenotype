from pathlib import Path
from argparse import ArgumentParser
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm


def acc_dice_cm(labels, targets):
    eps = 0.01
    cm = confusion_matrix(targets, labels)
    d = cm.diagonal()
    acc = (d.sum() + eps) / (cm.sum() + eps)
    tp = d.sum()
    dice = (2 * tp + eps) / (2 * tp + (cm - np.diag(d)).sum() + eps)
    return float(acc), float(dice), cm.astype(float).tolist()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--load_epoch', type=str, default='latest')
    parser.add_argument('--ihc_data_file', type=Path,
                        default='/well/rittscher/projects/IHC_Request/data/documents/additional_data_2020-04-21.csv')
    parser.add_argument('--split_path', type=Path, default='/well/rittscher/projects/IHC_Request/data/cross_validate/3-split2.json')
    args = parser.parse_args()
    results_dir = args.data_dir/'data'/'classifications'/f'{args.experiment_name}_{args.load_epoch}'
    classification_stats = {}
    slides_stats = classification_stats['slides_stats'] = {}
    slides_data = pd.read_csv(args.ihc_data_file)
    # measure classifier output on every slide
    slide_result_paths = list(results_dir.iterdir())
    with args.split_path.open('r') as split_path_file:
        split_path = json.load(split_path_file)
    test_split = set(split_path['test_slides'])
    for slide_result_path in tqdm(slide_result_paths):
        if slide_result_path.suffix != '.json':
            continue
        slide_id = slide_result_path.name[:-5]
        if slide_id not in test_split:
            continue
        slide_foci_data = slides_data[slides_data['Image'] == slide_id]
        if slide_foci_data.empty:
            continue
        if not np.isnan(slide_foci_data['Staining code'].iloc[0]):  # H&E slides have no stain label, hence they are read as nan
            continue
        case_type = slide_foci_data.iloc[0]['Case type']
        ihc_reasons = list(set(ihc_reason[0] if not isinstance(ihc_reason, float) else -1
                          for ihc_reason in slide_foci_data['IHC reason']))
        with slide_result_path.open('r') as slide_result_file:
            slide_results = json.load(slide_result_file)
        total_num_tiles = len(slide_results)
        num_ambiguous_tiles = sum(slide_result['classification'] for slide_result in slide_results)
        num_certain_tiles = total_num_tiles - num_ambiguous_tiles
        slides_stats[slide_id] = {
            'total_num_tiles': total_num_tiles,
            'num_certain_tiles': num_certain_tiles,
            'num_ambiguous_tiles': num_ambiguous_tiles,
            'certain_tiles_fraction': round(num_certain_tiles/total_num_tiles, 2),
            'ambiguous_tiles_fraction': round(num_ambiguous_tiles/total_num_tiles, 2),
            'case_type': case_type,
            'target': 1 if case_type == 'Real' else 0,
            'ihc_reasons': ihc_reasons,
            'label_by_majority_vote': int(num_ambiguous_tiles > num_certain_tiles),
            'average_certain_probability': float(np.mean([slide_result['probability_class0']
                                                            for slide_result in slide_results])),
            'average_ambiguous_probability': float(np.mean([slide_result['probability_class1']
                                                            for slide_result in slide_results]))
        }
    # evaluate classifier performance
    mean_total_num_tiles = np.median([slide_stat['total_num_tiles'] for slide_stat in slides_stats.values()])
    mean_num_certain_tiles = np.median([slide_stat['num_certain_tiles'] for slide_stat in slides_stats.values()])
    mean_num_ambiguous_tiles = np.median([slide_stat['num_ambiguous_tiles'] for slide_stat in slides_stats.values()])
    num_slides_no_ambiguous = sum([slide_stat['num_ambiguous_tiles'] == 0 for slide_stat in slides_stats.values()])
    median_certain_tile_fraction = np.median([slide_stat['certain_tiles_fraction']
                                              for slide_stat in slides_stats.values()])
    median_ambiguous_tile_fraction = np.median([slide_stat['ambiguous_tiles_fraction']
                                                for slide_stat in slides_stats.values()])
    labels_by_fraction = []
    for slide_id in slides_stats:
        # calculate a label by checking whether slide has more tiles belonging to
        label_by_fraction = int(slides_stats[slide_id]['ambiguous_tiles_fraction'] > median_ambiguous_tile_fraction)
        labels_by_fraction.append(label_by_fraction)
        slides_stats[slide_id]['label_by_fraction'] = label_by_fraction
    targets = [slide_stat['target'] for slide_stat in slides_stats.values()]
    labels_by_majority = [slide_stat['label_by_majority_vote'] for slide_stat in slides_stats.values()]
    accuracy_by_majority, dice_by_majority, confusion_matrix_by_majority = acc_dice_cm(labels_by_majority, targets)
    accuracy_by_fraction, dice_by_fraction, confusion_matrix_by_fraction = acc_dice_cm(labels_by_fraction, targets)
    classification_stats.update({
        'num_slides_no_ambiguous': num_slides_no_ambiguous,
        'mean_total_num_tiles': mean_total_num_tiles,
        'mean_num_certain_tiles': mean_num_certain_tiles,
        'mean_num_ambiguous_tile': mean_num_ambiguous_tiles,
        'median_certain_tile_fraction': median_certain_tile_fraction,
        'median_ambiguous_tile_fraction': median_ambiguous_tile_fraction,
        'accuracy_by_majority': accuracy_by_majority,
        'dice_by_majority': dice_by_majority,
        'confusion_matrix_by_majority': confusion_matrix_by_majority,
        'accuracy_by_fraction': accuracy_by_fraction,
        'dice_by_fraction': dice_by_fraction,
        'confusion_matrix_by_fraction': confusion_matrix_by_fraction
    })
    classification_results_path = args.data_dir/'data'/'classifications'/\
                                  f'{args.experiment_name}_slide_results_{str(datetime.now())[:10]}.json'
    with classification_results_path.open('w') as classification_results_file:
        json.dump(classification_stats, classification_results_file)
    print("Done!")




