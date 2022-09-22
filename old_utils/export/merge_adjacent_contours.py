import sys
sys.path.extend(['/well/rittscher/users/achatrian/cancer_phenotype/base',
                 '/well/rittscher/users/achatrian/cancer_phenotype'])
import argparse
from pathlib import Path
from annotation.annotation_path_merger import main as merge_annotation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotations_dir', required=True, type=Path)
    args, unparsed = parser.parse_known_args()
    annotation_paths = tuple(args.annotations_dir.iterdir())
    for i, annotation_path in enumerate(annotation_paths):
        if i > 0:
            sys.argv = sys.argv[:-1]  # remove path entry - cannot pop entries from argv
        print(f"Merging contours for '{annotation_path.name}'")
        sys.argv.append(f'--annotation_path={str(annotation_path)}')
        merge_annotation()
        print(f"Merged contours for {i+1} / {len(annotation_paths)} annotations")
