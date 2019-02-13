import sys
import pytest
import numpy as np
import openslide
import matplotlib.pyplot as plt
from itertools import product
sys.path.extend(['../', '../base/', '../base/options/*', '../base/data/*'])
import base.data as data
from base.utils import utils


def test_is_HnE(wsi_file):
    slide = openslide.open_slide(wsi_file)
    loc = [dim // 2 for dim in slide.level_dimensions[0]]
    tile0 = np.array(slide.read_region(loc, 0, (2048, 2048)))
    tile1 = np.array(slide.read_region(loc, 1, (1024, 1024)))
    tile2 = np.array(slide.read_region(loc, 2, (512, 512)))
    tile3 = np.array(slide.read_region(loc, 4, (256, 256)))
    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(tile0)
    axes[1].imshow(tile1)
    axes[2].imshow(tile2)
    axes[3].imshow(tile3)
    plt.show()
    is_hne0 = data.wsi_reader.is_HnE(tile0, slide.level_dimensions[0])
    is_hne1 = data.wsi_reader.is_HnE(tile1, slide.level_dimensions[0])
    is_hne2 = data.wsi_reader.is_HnE(tile2, slide.level_dimensions[0])
    is_hne3 = data.wsi_reader.is_HnE(tile3, slide.level_dimensions[0])
    assert is_hne0 and is_hne1 and is_hne2 and is_hne3


def test_wsi_reader(apply_options, wsi_file):
    assert utils.is_pathname_valid(wsi_file)
    sys.argv.extend(['--data_dir=/home/sedm5660/Documents/Temp/Data/cancer_phenotype']),
    sys.argv.extend(['--dataset_name=wsi'])
    opt = apply_options.parse()
    slide = data.wsi_reader.WSIReader(opt, wsi_file)
    slide.find_good_locations()
    assert len(slide) > 0
    print("{} tiles of shape {} passed quality control".format(len(slide), opt.patch_size))
    fig, axes = plt.subplots(3, 3)
    for (i, j), n in zip(product(range(0, 3), range(0, 3)), np.random.randint(0, len(slide), size=(9,))):
        tile = slide[n]
        axes[i, j].imshow(tile)
    plt.show()


def test_tcga_dataset(apply_options, tcga_data):
    import sys
    options = apply_options
    sys.argv.extend(['--dataset_name=tcga',
                     '--wsi_tablefile=/home/sedm5660/Documents/Temp/Data/cancer_phenotype/tcga_data_info/biospecimen.project-TCGA-PRAD.2018-10-05/sample.tsv',
                     '--cna_tablefile=/home/sedm5660/Documents/Temp/Data/cancer_phenotype/tcga_data_info/prad_tcga_pan_can_atlas_2018/data_CNA.txt'
                     ])
    opt = options.parse()
    tcga_dataset = data.create_dataset(opt)









