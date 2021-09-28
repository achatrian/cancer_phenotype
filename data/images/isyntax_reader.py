import os
import argparse
from pathlib import Path
import re
import warnings
from typing import Tuple
import csv
import json
from numbers import Real
import datetime
import time
from openslide import OpenSlideError
import tqdm
import numpy as np
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes
from .base_wsi_reader import IsyntaxReader as IReader
from base.utils import debug




# if __name__ == '__main__':
#     opt = IsyntaxReader.get_reader_options()
#     wsi_reader = IsyntaxReader(opt.slide_path, opt)
#     wsi_reader.find_tissue_locations(opt.tissue_threshold, opt.saturation_threshold)
#     wsi_reader.export_tissue_tiles('tiles_temp')








