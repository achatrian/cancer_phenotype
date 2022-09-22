import sys
import os
import numpy as np
import openslide
from matplotlib import pyplot as plt
from itertools import product
sys.path.extend(['../', '../base/', '../base/options/*', '../base/data/*'])
import base.datasets as data
from base.utils import utils
import errno

# Sadly, Python fails to provide the following magic number for us.
ERROR_INVALID_NAME = 123
'''
Windows-specific error code indicating an invalid pathname.

See Also
----------
https://msdn.microsoft.com/en-us/library/windows/desktop/ms681382%28v=vs.85%29.aspx
    Official listing of all such codes.
'''


def is_pathname_valid(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?


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
    is_hne0 = image.wsi_reader.is_HnE(tile0, slide.level_dimensions[0])
    is_hne1 = image.wsi_reader.is_HnE(tile1, slide.level_dimensions[0])
    is_hne2 = image.wsi_reader.is_HnE(tile2, slide.level_dimensions[0])
    is_hne3 = image.wsi_reader.is_HnE(tile3, slide.level_dimensions[0])
    assert is_hne0 and is_hne1 and is_hne2 and is_hne3


def test_wsi_reader(apply_options, wsi_file):
    assert utils.is_pathname_valid(wsi_file)
    sys.argv.extend(['--data_dir=/home/sedm5660/Documents/Temp/Data/cancer_phenotype']),
    sys.argv.extend(['--dataset_name=wsi'])
    opt = apply_options.parse()
    slide = image.wsi_reader.make_wsi_reader(opt, wsi_file)
    slide.find_tissue_locations()
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
    tcga_dataset = datasets.create_dataset(opt)









