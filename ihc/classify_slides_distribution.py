from pathlib import Path
from argparse import ArgumentParser
import bisect
import json
from pprint import pprint
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, plot_roc_curve
from matplotlib.pyplot import figure, axes
import joblib


r"""Find reduced dimensionality space for slide vectors, train slide classifier on train split and test it on test split"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--load_epoch', type=str, default='latest')
    parser.add_argument('--classifier_experiment', default=None)
    parser.add_argument('--representative_tiles', type=int, default=20)
    parser.add_argument('--ihc_data_file', type=Path,
                        default='/well/rittscher/projects/IHC_Request/data/documents/additional_data_2020-04-21.csv')
    parser.add_argument('--data_split', type=Path, default='/well/rittscher/projects/IHC_Request/data/cross_validate/3-split2.json')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num_estimators', default=200, type=int)
    args = parser.parse_args()
    with open(args.ihc_data_file, 'r') as ihc_data_file:
        slides_data = pd.read_csv(ihc_data_file)
    with args.data_split.open('r') as data_split_file:
        data_split = json.load(data_split_file)
    train_split, test_split = set(data_split['train_slides']), set(data_split['test_slides'])
    # read classification
    classification_dir = (args.data_dir/'data'/'classifications'/(args.experiment_name + f'_{args.load_epoch}'))
    # find top ambiguous tiles (scored by variance)
    slides_classifications = list(classification_dir.iterdir())
    # for every slide, find ambiguous tiles with top variance and build feature vector for classifier
    pca = IncrementalPCA(n_components=25)
    feature_buffer = []
    slides_vectors, labels = {}, {}
    slides_with_no_vector = []
    try:
        if args.overwrite:
            raise FileNotFoundError('overwrite')
        with open((args.data_dir/'data'/'classifications'/(f'vectors_{args.experiment_name}_{args.load_epoch}.json')), 'r') \
                as vectors_file:
            slides_vectors = json.load(vectors_file)
        with open((args.data_dir/'data'/'classifications'/(f'labels_{args.experiment_name}_{args.load_epoch}.json')), 'r') \
                as labels_file:
            labels = json.load(labels_file)
        if args.classifier_experiment is not None:
            with open((args.data_dir/'data'/'classifications'/(f'pca_{args.classifier_experiment}_{args.load_epoch}.joblib')), 'rb') \
                    as pca_file:  # NB only one 'load_epoch' parameter for both cases of if statement
                pca = joblib.load(pca_file)
            print(f"Feature space loaded from '{args.classifier_experiment}'")
        else:
            with open((args.data_dir/'data'/'classifications'/(f'pca_{args.experiment_name}_{args.load_epoch}.joblib')), 'rb') \
                    as xgbc_file:
                pca = joblib.load(xgbc_file)
    except (FileNotFoundError, json.JSONDecodeError) as err:
        print("Error in reading files:")
        print(err)
        print("Building slide vectors ... ")
        for classification_path in tqdm(slides_classifications, desc='slide'):
            if classification_path.suffix != '.json':
                continue
            slide_id = classification_path.with_suffix('').name
            slide_foci_data = slides_data[slides_data['Image'] == slide_id]
            if slide_foci_data.empty:
                continue
            if 'Staining code' in slide_foci_data.columns and \
                    not (lambda t: isinstance(t, float) and np.isnan(t))(slide_foci_data['Staining code'].iloc[0]):
                # H&E slides have no stain label, hence they are read as nan -- validation is all H&E (no staining code)
                continue
            if slide_id in slides_vectors:
                top_ambiguous_tiles = slides_vectors[slide_id]
            else:
                top_ambiguous_tiles = slides_vectors[slide_id] = [{
                    'variance': 0.0, 'loss_variance': 0.0,
                    'classification': 0,
                    'probability_class0': 1.0, 'probability_class1': 0.0,
                }]
            if 'Case type' in slide_foci_data.columns:
                case_type = slide_foci_data.iloc[0]['Case type']
                labels[slide_id] = 0 if case_type == 'Control' else 1
            elif 'IHC request' in slide_foci_data.columns:
                ihc_request = slide_foci_data.iloc[0]['IHC request']
                labels[slide_id] = 0 if ihc_request == 'No' else 1
            else:
                raise ValueError("Invalid slide data file format -- must have either a 'Case type' or 'IHC request' field")
            with classification_path.open('r') as classification_file:
                classification = json.load(classification_file)
            for tile_result in tqdm(classification, desc='tile'):
                feature_buffer.append(tile_result['features'])
                if len(feature_buffer) > 25:
                    pca = pca.partial_fit(np.array(feature_buffer))
                    feature_buffer.clear()
                if tile_result['classification'] == 1 and any(result['classification'] == 0 for result in top_ambiguous_tiles):
                    insertion_index = next(i for i, result in enumerate(reversed(top_ambiguous_tiles)) if result['classification'] == 0)
                    top_ambiguous_tiles.insert(insertion_index, tile_result)
                    print(len(top_ambiguous_tiles))  # TODO remove
                elif tile_result['variance'] > top_ambiguous_tiles[0]['variance']:
                    insertion_index = bisect.bisect_right([r['variance'] for r in top_ambiguous_tiles], tile_result['variance'])
                    top_ambiguous_tiles.insert(insertion_index, tile_result)
                    print(len(top_ambiguous_tiles))  # TODO remove
                if len(top_ambiguous_tiles) > args.representative_tiles:
                    del top_ambiguous_tiles[0]
            if top_ambiguous_tiles[0]['variance'] == 0.0:
                del top_ambiguous_tiles[0]
            if len(top_ambiguous_tiles) < args.representative_tiles:
                del slides_vectors[slide_id]
                slides_with_no_vector.append(slide_id)
                continue
            assert len(top_ambiguous_tiles) == args.representative_tiles, f"{args.representative_tiles} tiles are required"
            tqdm.write(f"Min variance: {top_ambiguous_tiles[0]['variance']}; max variance: {top_ambiguous_tiles[-1]['variance']}")
        print(f"{len(slides_with_no_vector)} slides did not have enough classified tiles")
        with open((args.data_dir/'data'/'classifications'/(f'vectors_{args.experiment_name}_{args.load_epoch}.json')), 'w') \
                as vectors_file:
            json.dump(slides_vectors, vectors_file)
        with open((args.data_dir/'data'/'classifications'/(f'labels_{args.experiment_name}_{args.load_epoch}.json')), 'w') \
                as labels_file:
            json.dump(labels, labels_file)
        with open((args.data_dir/'data'/'classifications'/(f'pca_{args.experiment_name}_{args.load_epoch}.joblib')), 'wb') \
                as pca_file:
            joblib.dump(pca, pca_file)
    # build data
    print("PCA explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Building slide signatures ... ")
    train_signatures, train_labels, test_signatures, test_labels = [], [], [], []
    for slide_id in slides_vectors:
        tiles_results = slides_vectors[slide_id]
        signature = []
        for tile_result in tiles_results:
            signature.append(tile_result['probability_class0'])
            signature.append(tile_result['probability_class1'])
            signature.append(tile_result['variance'])
            signature.append(tile_result['loss_variance'])
            signature.extend(pca.transform(np.array(tile_result['features'])[np.newaxis, ...]).squeeze().tolist())
        if slide_id in train_split:
            train_signatures.append(signature)
            train_labels.append(labels[slide_id])
        elif slide_id in test_split:
            test_signatures.append(signature)
            test_labels.append(labels[slide_id])
    # classify
    train_signatures, test_signatures, train_labels, test_labels = np.array(train_signatures), np.array(test_signatures), \
                                                  np.array(train_labels), np.array(test_labels)
    try:
        if args.overwrite:
            raise FileNotFoundError('overwrite')
        if args.classifier_experiment is not None:
            with open((args.data_dir / 'data' / 'classifications' / (f'xgbc_{args.classifier_experiment}_{args.load_epoch}.joblib')), 'rb') \
                    as xgbc_file:  # NB only one 'load_epoch' parameter for both cases of if statement
                classifier = joblib.load(xgbc_file)
            print(f"Classifier loaded from '{args.classifier_experiment}'")
        else:
            with open((args.data_dir / 'data' / 'classifications' / (f'xgbc_{args.experiment_name}_{args.load_epoch}.joblib')), 'rb') \
                    as xgbc_file:
                classifier = joblib.load(xgbc_file)
    except FileNotFoundError:
        classifier = XGBClassifier(n_estimators=args.num_estimators, max_depth=40, subsample=0.9)
        classifier.fit(train_signatures, train_labels)
        with open((args.data_dir / 'data' / 'classifications' / (f'xgbc_{args.experiment_name}_{args.load_epoch}.joblib')), 'wb') \
                as xgbc_file:  # NB only one 'load_epoch' parameter for both cases of if statement
            joblib.dump(classifier, xgbc_file)
    predicted_labels = classifier.predict(test_signatures)
    predicted_probabilities = classifier.predict_proba(test_signatures)
    accuracy = (np.array(test_labels) == np.array(predicted_labels)).mean()
    auc = roc_auc_score(test_labels, predicted_probabilities[:, 1])  # probability for positive class
    with open(args.data_dir/'data'/'classifications'/(f'rfc_{args.experiment_name}_{args.load_epoch}.json'), 'w') \
        as random_forest_class_file:
        results = {'accuracy': accuracy, 'auc': auc, 'train_size': len(train_signatures), 'test_size': len(test_signatures)}
        results.update({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        json.dump(results, random_forest_class_file)
    pprint(results)
    fig = figure()
    ax = axes()
    plot_roc_curve(classifier, test_signatures, test_labels, ax=ax)
    fig.savefig(args.data_dir/'data'/'classifications'/(f'roc_{args.experiment_name}_{args.load_epoch}.png'))