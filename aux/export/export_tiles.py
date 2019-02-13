import pickle
import sys
from os.path import basename as os_basename
sys.path.extend([
    '/well/rittscher/users/achatrian/cancer_phenotype/',
    '/well/rittscher/users/achatrian/cancer_phenotype/base'
])
from base.data.wsi_reader import WSIReader


def main(slide_file, opt_file):
    print(f"Py Processing {os_basename(slide_file)}")
    opt = pickle.load(open(opt_file, 'rb'))
    setattr(opt, 'overwrite_qc', True)  # force overwriting of all quality_control files
    print(f"Quality control mpp: {opt.qc_mpp}, read_mpp: {opt.mpp}")
    slide = WSIReader(opt, slide_file)
    slide.find_good_locations()
    slide.export_good_tiles('tiles2')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])  # first entry in argv is name of file


