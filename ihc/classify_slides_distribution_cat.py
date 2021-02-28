from pathlib import Path
from argparse import ArgumentParser
import json
from pprint import pprint
import pickle
from tqdm import tqdm
import numpy as np
from scipy.stats import kurtosis
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, plot_roc_curve
from matplotlib.pyplot import figure, axes
import joblib


r"""Find reduced dimensionality space for slide vectors, 
train slide classifier on train split and test it on test split,
difference from classify slides script is that here classification is based on the statistics of tile signature distributions
"""


def grid_search_on_random_forest(train_signatures, train_labels, test_signatures, test_labels, model_dir=None, classifiers=None):
    classifiers = classifiers or []
    best_classifier = None
    best_auc = 0.0
    print("len of classifiers", len(classifiers))
    for num_estimators in tqdm([50, 100, 150], desc="num trees", position=0):
        for max_depth in tqdm([10, 16], desc="max depth", position=1):
            for subsample in tqdm([0.7, 0.9], desc="subsample", position=2):
                parameters = f'N:{num_estimators},D:{max_depth},S:{subsample}'
                try:
                    classifier = next(classifier for classifier in classifiers
                                      if classifier['parameters'] == parameters)['classifier']
                except StopIteration:
                    classifier = CatBoostClassifier(num_trees=num_estimators, max_depth=max_depth, subsample=subsample)
                    classifier.fit(train_signatures, train_labels)
                    if model_dir is not None:
                        classifier.save_model(str(model_dir/f'cat_{parameters}.cbm'))
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


def load_models(experiment_model_dir):
    classifiers = []
    for file_path in experiment_model_dir.iterdir():
        if not file_path.name.startswith('cat'):
            continue
        parameters = file_path.with_suffix('').name.split('_')[1]
        model = CatBoostClassifier()
        model.load_model(str(file_path))
        classifiers.append({
            'parameters': parameters,
            'classifier': model,
            'results': {}
        })
    return classifiers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--load_epoch', type=str, default='latest')
    parser.add_argument('--classifier_experiment', default=None)
    parser.add_argument('--num_components', type=int, default=25)
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
    experiment_model_dir = classification_dir.parent / f'model_{args.experiment_name}_{args.load_epoch}'
    experiment_model_dir.mkdir(exist_ok=True, parents=True)
    if args.classifier_experiment:
        classifier_experiment_model_dir = classification_dir.parent / f'model_{args.classifier_experiment}_{args.load_epoch}'
    else:
        classifier_experiment_model_dir = None
    # find top ambiguous tiles (scored by variance)
    slides_classifications = list(classification_dir.iterdir())
    # for every slide, find ambiguous tiles with top variance and build feature vector for classifier
    pca = IncrementalPCA(n_components=args.num_components)
    feature_buffer = []
    slides_vectors, labels = {}, {}
    slides_with_no_vector = []
    buffer = None
    try:
        print("Loading slide vectors and labels ...")
        if args.overwrite:
            raise FileNotFoundError("overwrite vectors ...")
        with open(experiment_model_dir / f'slides_vectors.joblib', 'rb') as slides_vectors_file:
            slides_vectors = joblib.load(slides_vectors_file)
        with open(experiment_model_dir / f'slides_labels.joblib', 'rb') as labels_file:
            labels = joblib.load(labels_file)
        with open(experiment_model_dir / f'features_pca.joblib', 'rb') as pca_file:
            pca = joblib.load(pca_file)
    except FileNotFoundError as err:
        print(err)
        print("Building slide vectors and labels ...")
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
        with open(experiment_model_dir / f'slides_vectors.joblib', 'wb') as slides_vectors_file:
            joblib.dump(slides_vectors, slides_vectors_file)
        with open(experiment_model_dir / f'slides_labels.joblib', 'wb') as labels_file:
            joblib.dump(labels, labels_file)
        with open(experiment_model_dir / f'features_pca.joblib', 'wb') as pca_file:
            joblib.dump(pca, pca_file)
    # build data
    if args.classifier_experiment is not None:
        with open(classifier_experiment_model_dir / f'features_pca.joblib', 'rb') as pca_file:
            # NB only one 'load_epoch' parameter for both cases of if statement
            pca = joblib.load(pca_file)
        print(f"Feature space loaded from '{args.classifier_experiment}'")
    print("PCA explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Building slide signatures ... ")
    train_signatures, train_labels, test_signatures, test_labels = [], [], [], []
    for slide_id in tqdm(slides_vectors, desc='building signatures ...'):
        tiles_results = slides_vectors[slide_id]
        ambiguous_features = np.array([tile_result['features'] for tile_result in tiles_results['ambiguous_tiles']])
        ambiguous_features = pca.transform(ambiguous_features)
        certain_features = np.array([tile_result['features'] for tile_result in tiles_results['certain_tiles']])
        certain_features = pca.transform(certain_features)
        statistics = [
            np.median(ambiguous_features, axis=0),
            np.mean(ambiguous_features, axis=0),
            np.std(ambiguous_features, axis=0),
            kurtosis(ambiguous_features),
            np.median(certain_features, axis=0),
            np.mean(certain_features, axis=0),
            np.std(certain_features, axis=0),
            kurtosis(certain_features),
            [np.median([tile_result['variance'] for tile_result in tiles_results['ambiguous_tiles']])],
            [np.median([tile_result['variance'] for tile_result in tiles_results['certain_tiles']])],
            [np.median([tile_result['loss_variance'] for tile_result in tiles_results['ambiguous_tiles']])],
            [np.median([tile_result['loss_variance'] for tile_result in tiles_results['certain_tiles']])],
            [len(ambiguous_features), len(certain_features)],
        ]
        signature = np.concatenate(statistics)
        if slide_id in train_split:
            train_signatures.append(signature)
            train_labels.append(labels[slide_id])
        elif slide_id in test_split:
            test_signatures.append(signature)
            test_labels.append(labels[slide_id])
    # classify
    train_signatures, test_signatures, train_labels, test_labels = np.array(train_signatures), np.array(test_signatures), \
                                                  np.array(train_labels), np.array(test_labels)
    pprint(train_signatures)
    experiment_model_dir.mkdir(exist_ok=True, parents=True)
    with open(experiment_model_dir / f'signatures.json', 'w') as signatures_file:
        json.dump({
            'train_signatures': train_signatures.tolist(), 'test_signatures': test_signatures.tolist(),
            'train_labels': train_labels.tolist(), 'test_labels': test_labels.tolist()
        }, signatures_file)
    print("Normalizing signatures ...")
    if args.classifier_experiment is not None:
        with open(classifier_experiment_model_dir / f'signatures_scaler.joblib', 'rb') as scaler_file:
            scaler = joblib.load(scaler_file)
    else:
        scaler = MinMaxScaler()
        scaler.fit(np.concatenate((train_signatures, test_signatures)))
    if train_signatures.size > 0:
        train_signatures = scaler.transform(train_signatures)
    test_signatures = scaler.transform(test_signatures)
    with open(experiment_model_dir / f'signatures_scaler.joblib', 'wb') as scaler_file:
        joblib.dump(scaler, scaler_file)
    try:
        if args.overwrite:
            raise FileNotFoundError('overwrite')
        if args.classifier_experiment is not None:
            classifiers = load_models(classifier_experiment_model_dir)
            print(f"Classifier loaded from '{args.classifier_experiment}'")
        else:
            classifiers = load_models(experiment_model_dir)
            print("Classifier was loaded ")
        classifier, classifiers = grid_search_on_random_forest(train_signatures, train_labels,
                                                               test_signatures, test_labels,
                                                               experiment_model_dir, classifiers)
    except FileNotFoundError:
        print("Training classifier ...")
        classifier, classifiers = grid_search_on_random_forest(train_signatures, train_labels, test_signatures, test_labels)
    grid_search_results = {}
    # save all models and results
    for iteration in classifiers:
        parameters, model, results = tuple(iteration.values())
        grid_search_results[parameters] = results
    predicted_labels = classifier.predict(test_signatures)
    predicted_probabilities = classifier.predict_proba(test_signatures)
    accuracy = (np.array(test_labels) == np.array(predicted_labels)).mean()
    auc = roc_auc_score(test_labels, predicted_probabilities[:, 1])  # probability for positive class
    with open(experiment_model_dir/f'results.json', 'w') as results_file:
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'train_size': len(train_signatures),
            'test_size': len(test_signatures),
            'grid_search_results': grid_search_results
        }
        results.update({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        json.dump(results, results_file)
    fig = figure()
    ax = axes()
    pprint(results)
    plot_roc_curve(classifier, test_signatures, test_labels, ax=ax)
    fig.savefig(experiment_model_dir/(f'best_classifier_roc.png'))