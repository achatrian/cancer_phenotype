from pathlib import Path
import pandas as pd
from tqdm import tqdm

r"""Delete selected segmentation slides"""


if __name__ == '__main__':
    master_list_path = Path('/mnt/rescomp/projects/ProMPT/cases/ProMPT_master_list.xlsx')
    slides_list = pd.read_excel(master_list_path, sheet_name='slides')
    selected_cases_path = Path('/mnt/rescomp/users/achatrian/ProMPT_biopsy_vs_rp_cases_sample_merged.xlsx')
    selected_cases = pd.read_excel(selected_cases_path)
    selected_slides_list = slides_list[slides_list['SpecimenIdentifier'].isin(selected_cases['SpecimenIdentifier'])]
    masks_dir = Path('/mnt/rescomp/projects/ProMPT/data/masks/combined_mpp1.0_normal')
    prob_map_dir0 = masks_dir/'prob_maps'
    prob_map_dir1 = masks_dir/'prob_maps_0.5shift'
    annotations_dir = Path('/mnt/rescomp/projects/ProMPT/data/annotations/combined_mpp1.0_normal')
    # xl_slides = slides_list[slides_list['XL'] == 1]
    removed_prob_maps0, removed_prob_maps1, removed_masks, removed_annotations = 0, 0, 0, 0
    for slide_id in tqdm(slides_list['SlideIdentifier']):
        # try:
        #     prob_map_path0 = prob_map_dir0/(slide_id + '.tiff')
        #     prob_map_path0.unlink()
        #     removed_prob_maps0 += 1
        # except FileNotFoundError:
        #     pass
        # try:
        #     prob_map_path1 = prob_map_dir1/(slide_id + '.tiff')
        #     prob_map_path1.unlink()
        #     removed_prob_maps1 += 1
        # except FileNotFoundError:
        #     pass
        try:
            mask_path = masks_dir/(slide_id + '.tiff')
            mask_path.unlink()
            removed_masks += 1
        except FileNotFoundError:
            pass
        try:
            annotation_path = annotations_dir/(slide_id + '.json')
            annotation_path.unlink()
            removed_annotations += 1
        except FileNotFoundError:
            pass
    print(f"(XL) Removed {removed_prob_maps0} maps, {removed_prob_maps1} shifted maps, {removed_annotations} annotations, and {removed_masks} masks")

