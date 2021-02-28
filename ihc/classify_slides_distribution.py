from pathlib import Path
from argparse import ArgumentParser
import json
from pprint import pprint
import pickle
from collections import Counter
from tqdm import tqdm
import numpy as np
from scipy.stats import kurtosis
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
# import xgboost.core.XGBoostError
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix
from matplotlib.pyplot import figure, axes
import joblib


r"""Find reduced dimensionality space for slide vectors,
train slide classifier on train split and test it on test split,
difference from classify slides script is that here classification is based on the statistics of tile signature distributions
"""


def grid_search_on_random_forest(train_signatures, train_labels, test_signatures, test_labels, classifiers=None,
                                 use_weight_scale=True):
    classifiers = classifiers or []
    best_classifier = None
    best_auc = 0.0
    print("len of classifiers", len(classifiers))
    for num_estimators in [200, 400, 600, 800]:
        for max_depth in [20, 40, 60]:
            for subsample in [0.5, 0.7, 0.9]:
                if len(classifiers) != 4*3*3:
                    label_counter = Counter(train_labels)
                    weight = label_counter[0] / label_counter[1]
                    classifier = XGBClassifier(
                        n_estimators=num_estimators,
                        max_depth=max_depth,
                        subsample=subsample,
                        scale_pos_weight=weight if use_weight_scale else None,
                        seed=42
                    )
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


def load_models(experiment_model_dir):
    classifiers = []
    for file_path in experiment_model_dir.iterdir():
        if not file_path.name.startswith('xgbc'):
            continue
        parameters = file_path.with_suffix('').name.split('_')[1]
        try:
            with open(experiment_model_dir / f'xgbc_{parameters}.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
        except BaseException:
            model = XGBClassifier()
            model.load_model(str(experiment_model_dir / f'comp_xgbc_{parameters}.pkl'))
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
    parser.add_argument('--annotator_name', type=str, default=None)
    args = parser.parse_args()
    with open(args.ihc_data_file, 'r') as ihc_data_file:
        slides_data = pd.read_csv(ihc_data_file)
    with args.data_split.open('r') as data_split_file:
        data_split = json.load(data_split_file)
    train_split, test_split = set(data_split['train_slides']), set(data_split['test_slides'])
    # read classification
    classification_dir = (args.data_dir/'data'/'classifications'/(args.experiment_name + f'_{args.load_epoch}'))
    experiment_model_dir = classification_dir.parent / f'model_{args.experiment_name}_{args.load_epoch}_{args.annotator_name}'
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
    discarded_slides = {}
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
        print(f"DNN features are available for {len(slides_classifications)} slides")
        for classification_path in tqdm(slides_classifications, desc='slide'):
            if classification_path.suffix != '.json':
                continue
            slide_id = classification_path.with_suffix('').name
            slide_foci_data = slides_data[slides_data['Image'] == slide_id]
            if slide_foci_data.empty:
                discarded_slides[slide_id] = 'absent_from_database'
                continue
            if 'Staining code' in slide_foci_data.columns and \
                    not (lambda t: isinstance(t, float) and np.isnan(t))(slide_foci_data['Staining code'].iloc[0]):
                # H&E slides have no stain label, hence they are read as nan -- validation is all H&E (no staining code)
                discarded_slides[slide_id] = 'not_h&e'
                continue
            if 'Case type' in slide_foci_data.columns:
                case_type = slide_foci_data.iloc[0]['Case type']
                if case_type == 'Control':
                    labels[slide_id] = 0
                elif case_type == 'Real':
                    labels[slide_id] = 1
                else:
                    discarded_slides[slide_id] = 'missing_label'
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
                discarded_slides[slide_id] = 'whole_slide_is_ambiguous_(outlier)'
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
        with open(experiment_model_dir / 'slides_vectors.joblib', 'wb') as slides_vectors_file:
            joblib.dump(slides_vectors, slides_vectors_file)
        with open(experiment_model_dir / 'slides_labels.joblib', 'wb') as labels_file:
            joblib.dump(labels, labels_file)
        with open(experiment_model_dir / 'features_pca.joblib', 'wb') as pca_file:
            joblib.dump(pca, pca_file)
        with open(experiment_model_dir / 'discarded_slides.json', 'w') as discarded_slides_file:
            json.dump(discarded_slides, discarded_slides_file)
    with open(experiment_model_dir/'slides_ids.json', 'w') as slides_ids_file:
        json.dump(list(slides_vectors.keys()), slides_ids_file)
    # build data
    print(f"Total slides: {len(labels)}, IHC-ordered: {sum(labels.values())}, Control: {len(labels) - sum(labels.values())}")
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
    # augment data
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
                                                               test_signatures, test_labels, classifiers)
    except FileNotFoundError:
        print("Training classifier ...")
        classifier, classifiers = grid_search_on_random_forest(train_signatures, train_labels, test_signatures, test_labels)
    grid_search_results = {}
    # save all models and results
    for iteration in classifiers:
        parameters, model, results = tuple(iteration.values())
        if (not (experiment_model_dir / f'xgbc_{parameters}.pkl').exists() and args.classifier_experiment is None) \
                or args.overwrite:
            # with open(experiment_model_dir / f'xgbc_{parameters}.joblib', 'wb') as model_file:
            #     joblib.dump(model, model_file)
            with open(experiment_model_dir / f'xgbc_{parameters}.pkl', 'wb') as model_file:
                pickle.dump(model, model_file)
            model.save_model(str(experiment_model_dir / f'comp_xgbc_{parameters}.bin'))
        grid_search_results[parameters] = results
    predicted_labels = classifier.predict(test_signatures)
    predicted_probabilities = classifier.predict_proba(test_signatures)
    accuracy = (np.array(test_labels) == np.array(predicted_labels)).mean()
    auc = roc_auc_score(test_labels, predicted_probabilities[:, 1])  # probability for positive class
    tn, fp, fn, tp = confusion_matrix(test_labels, predicted_labels).ravel()
    misclassified = [slide_id for slide_id, prediction, target
                     in zip(slides_vectors.keys(), predicted_labels, test_labels) if prediction != target]
    with open(experiment_model_dir/f'results.json', 'w') as results_file:#
        results = {
            'accuracy': accuracy,
            'auc': auc,
            'specificity': tn/(tn + fp),
            'sensitivity': tp/(tp + fn),
            'precision': tp/(tp + fp),
            'train_size': len(train_signatures),
            'test_size': len(test_signatures),
            'grid_search_results': grid_search_results,
            'misclassified': misclassified,
            'discarded_slides': discarded_slides,
            'predicted_labels': predicted_labels.tolist(),
            'test_labels': test_labels.tolist(),
            'predicted_probabilities': predicted_probabilities.tolist()
        }
        results.update({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        json.dump(results, results_file)
    pprint(results)
    fig = figure()
    ax = axes()

    def set_props(title, xlabel=None, ylabel=None, ax=None):
        ax = ax or plt.gca()
        ax.set_title(title, fontsize=20)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=16)
        else:
            ax.xaxis.label.set_size(16)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=16)
        else:
            ax.yaxis.label.set_size(16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    plot_roc_curve(classifier, test_signatures, test_labels, ax=ax)
    set_props('ROC curve')
    fig.savefig(experiment_model_dir/(f'best_classifier_roc.png'))
