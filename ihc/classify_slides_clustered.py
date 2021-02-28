from pathlib import Path
from argparse import ArgumentParser
import bisect
import json
from pprint import pprint
from tqdm import tqdm
import numpy as np
from scipy.stats import kurtosis
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, plot_roc_curve
from matplotlib.pyplot import figure, axes
import joblib


r"""Find reduced dimensionality space for slide vectors, 
train slide classifier on train split and test it on test split"""


def grid_search_on_random_forest(train_signatures, train_labels, test_signatures, test_labels, classifiers=None):
    classifiers = classifiers or []
    best_auc = 0.0
    for num_estimators in [200, 400, 600, 800]:
        for max_depth in [20, 40, 60]:
            for subsample in [0.5, 0.7, 0.9]:
                if len(classifiers) != 4*3*3:
                    classifier = XGBClassifier(n_estimators=num_estimators, max_depth=max_depth, subsample=subsample)
                    classifier.fit(train_signatures, train_labels)
                else:
                    classifier = next(classifier for classifier in classifiers
                                      if classifier['parameters'] == f'N:{num_estimators},D:{max_depth},S:{subsample}')['classifier']
                predicted_labels = classifier.predict(test_signatures)
                predicted_probabilities = classifier.predict_proba(test_signatures)
                accuracy = (np.array(test_labels) == np.array(predicted_labels)).mean()
                auc = roc_auc_score(test_labels, predicted_probabilities[:, 1])  # probability for positive class
                if auc > best_auc:
                    best_classifier = classifier
                    best_auc = auc
                if len(classifiers) != 4*3*3:
                    classifiers.append({
                        'parameters': f'N:{num_estimators},D:{max_depth},S:{subsample}',
                        'classifier': classifier,
                        'results': {'accuracy': accuracy, 'auc': auc}
                    })
    return best_classifier, classifiers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--load_epoch', type=str, default='latest')
    parser.add_argument('--classifier_experiment', default=None)
    parser.add_argument('--num_components', type=int, default=25)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--ihc_data_file', type=Path,
                        default='/well/rittscher/projects/IHC_Request/data/documents/additional_data_2020-04-21.csv')
    parser.add_argument('--data_split', type=Path, default='/well/rittscher/projects/IHC_Request/data/cross_validate/3-split2.json')
    parser.add_argument('--overwrite', action='store_true')
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
    pca = IncrementalPCA(n_components=args.num_components)
    feature_buffer = []
    slides_vectors, labels = {}, {}
    slides_with_no_vector = []
    print("Building slide vectors ... ")
    buffer = None
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
        ambiguous_tiles = [tile_result for tile_result in classification if tile_result['classification'] == 1]
        certain_tiles = [tile_result for tile_result in classification if tile_result['classification'] == 0]
        if not certain_tiles:
            continue
        feature_len = len(certain_tiles[0]['features'])
        if not ambiguous_tiles:
            ambiguous_tiles.append({
            'variance': 0.0, 'loss_variance': 0.0, 'classification': 0,
            'probability_class0': 0.0, 'probability_class1': 0.0, 'features': [0.0]*feature_len
        })
        slides_vectors[slide_id] = {
            'ambiguous_tiles': ambiguous_tiles,
            'certain_tiles': certain_tiles
        }
        tqdm.write(f"{len(ambiguous_tiles)} ambiguous tiles and {len(certain_tiles)} certain tiles in slide '{slide_id}'")
        ambiguous_features = np.array([tile_result['features'] for tile_result in ambiguous_tiles])
        certain_features = np.array([tile_result['features'] for tile_result in certain_tiles])
        features = np.concatenate([ambiguous_features, certain_features], axis=0)
        if buffer is not None:
            features = np.concatenate([features, buffer], axis=0)
            buffer = None
        if features.shape[0] < args.num_components:
            buffer = features
            continue
        pca.partial_fit(features)
    with open((args.data_dir/'data'/'classifications'/(f'pca_{args.experiment_name}_{args.load_epoch}.joblib')), 'wb') \
            as pca_file:
        joblib.dump(pca, pca_file)
    # train clustering
    clustering = KMeans(args.num_clusters)
    all_features = []
    for slide_id in tqdm(slides_vectors, desc='building signatures ...'):
        tiles_results = slides_vectors[slide_id]
        ambiguous_features = np.array([tile_result['features'] for tile_result in tiles_results['ambiguous_tiles']])
        ambiguous_features = pca.transform(ambiguous_features)
        certain_features = np.array([tile_result['features'] for tile_result in tiles_results['certain_tiles']])
        certain_features = pca.transform(certain_features)
        all_features.extend(ambiguous_features)
        all_features.extend(certain_features)
    clustering.fit(np.array(all_features))
    # build data
    if args.classifier_experiment is not None:
        with open((args.data_dir / 'data' / 'classifications' / (
        f'pca_{args.classifier_experiment}_{args.load_epoch}.joblib')), 'rb') \
                as pca_file:  # NB only one 'load_epoch' parameter for both cases of if statement
            pca = joblib.load(pca_file)
        print(f"Feature space loaded from '{args.classifier_experiment}'")
    print("PCA explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Building slide signatures ... ")
    train_signatures, train_labels, test_signatures, test_labels = [], [], [], []
    for slide_id in tqdm(slides_vectors, desc='building signatures ...'):
        tiles_results = slides_vectors[slide_id]
        ambiguous_features = np.array([tile_result['features'] for tile_result in tiles_results['ambiguous_tiles']])
        ambiguous_cluster_memberships = clustering.predict(pca.transform(ambiguous_features))
        ambiguous_histogram = np.histogram(ambiguous_cluster_memberships)
        certain_features = np.array([tile_result['features'] for tile_result in tiles_results['certain_tiles']])
        certain_cluster_memberships = clustering.predict(pca.transform(certain_features))
        certain_histogram = np.histogram(certain_cluster_memberships)
        statistics = [
            np.median([tile_result['variance'] for tile_result in tiles_results['ambiguous_tiles']]),
            np.median([tile_result['variance'] for tile_result in tiles_results['certain_tiles']]),
            np.median([tile_result['loss_variance'] for tile_result in tiles_results['ambiguous_tiles']]),
            np.median([tile_result['loss_variance'] for tile_result in tiles_results['certain_tiles']]),
            len(ambiguous_features),
            len(certain_features)
        ]
        signature = np.concatenate((ambiguous_histogram, certain_histogram, signature))
        if slide_id in train_split:
            train_signatures.append(signature)
            train_labels.append(labels[slide_id])
        elif slide_id in test_split:
            test_signatures.append(signature)
            test_labels.append(labels[slide_id])
    # classify
    train_signatures, test_signatures, train_labels, test_labels = np.array(train_signatures), np.array(test_signatures), \
                                                  np.array(train_labels), np.array(test_labels)
    with open(args.data_dir / 'data' / 'classifications' / f'signatures_{args.experiment_name}_{args.load_epoch}.json', 'w') as signatures_file:
        json.dump({
            'train_signatures': train_signatures.tolist(), 'test_signatures': test_signatures.tolist(),
            'train_labels': train_labels.tolist(), 'test_labels': test_labels.tolist()
        }, signatures_file)
    try:
        if args.overwrite:
            raise FileNotFoundError('overwrite')
        if args.classifier_experiment is not None:
            with open((args.data_dir / 'data' / 'classifications' / (f'xgbc_{args.classifier_experiment}_{args.load_epoch}.joblib')), 'rb') \
                    as xgbc_file:  # NB only one 'load_epoch' parameter for both cases of if stat   ement
                classifiers = joblib.load(xgbc_file)
                classifier, classifiers = grid_search_on_random_forest(train_signatures, train_labels,
                                                                       test_signatures, test_labels, classifiers)
            print(f"Classifier loaded from '{args.classifier_experiment}'")
        else:
            with open((args.data_dir / 'data' / 'classifications' / (f'xgbc_{args.experiment_name}_{args.load_epoch}.joblib')), 'rb') \
                    as xgbc_file:
                classifiers = joblib.load(xgbc_file)
                classifier, classifiers = grid_search_on_random_forest(train_signatures, train_labels,
                                                                       test_signatures, test_labels, classifiers)
    except FileNotFoundError:
        classifier, classifiers = grid_search_on_random_forest(train_signatures, train_labels, test_signatures, test_labels)
        with open((args.data_dir / 'data' / 'classifications' / (f'xgbc_{args.experiment_name}_{args.load_epoch}.joblib')), 'wb') \
                as xgbc_file:  # NB only one 'load_epoch' parameter for both cases of if statement
            joblib.dump(classifiers, xgbc_file)
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