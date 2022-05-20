# cancer phenotype

The 'base' directory contains the scripts and utility objects to train, evaluate and deploy neural networks.
It can load models and datasets from other task folders, such as segment.

The 'quant' folder contains scripts to extract features from tissue components in images and cluster the examples.

The 'data' and 'annotation' directories contain scripts to process images, contours and annotations made on AIDA. Example applicatiosn are reading tiles from a slide, finding overlapping contours on a slide, and creating an annotation with those contours.
