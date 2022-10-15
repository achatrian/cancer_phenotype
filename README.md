# cancer phenotype

The 'base' directory contains the scripts and utility objects to train, evaluate and deploy neural networks.
It can load models and datasets from other task folders, such as segment.

The 'quant' folder contains scripts to extract features from tissue components in images and cluster the examples.

The 'data' and 'annotation' directories contain scripts to process images, contours and annotations made on AIDA. Example applicatiosn are reading tiles from a slide, finding overlapping contours on a slide, and creating an annotation with those contours.

# Options

The 'base' directory contains the base objects for the main classes that make up
the network training pipeline, i.e. 'options', 'datasets', and 'models'.
- 'options' contains the `BaseOptions` object which sets up the training environment
through the values set by command line argparse arguments.
It also contains functions to load task specific options,
which are stored in each task directory. *Task directories* are any directory in
the same folder as 'base', which contain the subfolders 'options', 'datasets', and 'models',
e.g. *segment* and *ihc*. The task folder is selected by the argument `--task`.
- The final set of arguments availables is determined by the task options + model options + dataset options.
The two latter options are set in the methods `modify_commandline_options` inherited
from the `BaseModel` and `BaseDataset` base classes.

# Models location
Models are saved / stored in the directory specified by the `--checkpoints_dir` argument.
Typically this was specified as `/well/rescomp/users/achatrian/experiments` .
Each experiment folder contains the logs for the training of the experiment,
including a JSON file containing all the arguments values that were specified to train the model 'opt.json'.
If one wishes to see which arguments have have been changed from their default value,
the file 'opt.txt' in the experiment folder contains this information.

# Relevant functionalities
The code was used across my PhD and not all of it is needed.
# Segmentation
## Inference
The script used to apply the segmentation model to a slide dataset is: `cancer_phenotype/base/process_slides.py`.
The script code to compile the dataset information, it then uses the `apply` method of `WSIProcessor` from
`cancer_phenotype/base/inference/wsi_processor.py` to apply the segmentation model to the slides.
The script uses the additional options `ProcesSlidesOptions` in `cancer_phenotype/base/ProcesSlidesOptions`.
The dataset directory is specified with `--data_dir`. Slides within subfolders will
be detected as long as the suffix is specified with `--image_suffix`. For thin biopsies
one needs to specify the boundaries where the model will be applied. The masks produced by
the tissue mask model trained by SM KS and MH are suitable, which should be located in the directory
specified by `--tissue_mask_dirname`. If `--no_tissue_mask` is specified the algorithm to find
histology boundary from the saturation value developed by KHT is used (method `find_HnE` of `cancer_phenotype/data/images/wsi_reader/WSIReader`).

Pathologists' tumour annotations are stored within `--area_annotation_dir`.
If `--extract_contours` is set, the annotation is used to select the area where the
segmentation mask produced by the network will be converted into paths
for calculation of features and visualisation in AIDA.

# Feature Calculation
The scripts to calculate features are stored in `cancer_phenotype/quant/quantify`.
