from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm

r"""Delete selected segmentation slides"""


if __name__ == '__main__':
    # master_list_path = Path('/mnt/rescomp/projects/ProMPT/cases/ProMPT_master_list.xlsx')
    # slides_list = pd.read_excel(master_list_path, sheet_name='slides')
    # selected_cases_path = Path('/mnt/rescomp/users/achatrian/ProMPT_biopsy_vs_rp_cases_sample_merged.xlsx')
    # selected_cases = pd.read_excel(selected_cases_path)
    # selected_slides_list = slides_list[slides_list['SpecimenIdentifier'].isin(selected_cases['SpecimenIdentifier'])]
    masks_dir = Path('/mnt/rescomp/projects/ProMPT/data/masks/combined_mpp1.0_normal')
    prob_map_dir0 = masks_dir/'prob_maps'
    prob_map_dir1 = masks_dir/'prob_maps_0.5shift'
    annotations_dir = Path('/mnt/rescomp/projects/ProMPT/data/annotations/combined_mpp1.0_normal')
    stain_matrices_dir = Path('/mnt/rescomp/projects/ProMPT/data/epithelium:stain_references/combined_mpp1.0_normal')
    stain_references_dir = stain_matrices_dir/'references'
    check_list_path = Path('/mnt/rescomp/projects/ProMPT/data/documents/prompt_segmentation_check.xlsx')
    check_list = pd.read_excel(check_list_path)
    # xl_slides = slides_list[slides_list['XL'] == 1]
    removed_prob_maps0, removed_prob_maps1, removed_masks, removed_annotations, removed_stain_references = 0, 0, 0, 0, 0
    (masks_dir/'old_10_12_21').mkdir(exist_ok=True)
    for slide_id, quality in tqdm(zip(check_list['SlideIdentifier'], check_list['Quality_10_12_21'])):
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
        slide_id = re.sub(r'\.(ndpi|svs|tiff|isyntax)', '', slide_id)
        try:
            if quality == 'x':
                mask_path = masks_dir/(slide_id + '.tiff')
                mask_path = mask_path.rename(mask_path.parent/'old_10_12_21'/(slide_id + '_old4.tiff'))
                removed_masks += 1
                try:
                    annotation_path = annotations_dir/(slide_id + '.json')
                    annotation_path.unlink()
                    removed_annotations += 1
                except FileNotFoundError:
                    pass
                try:
                    stain_matrix_path = stain_matrices_dir/(slide_id + '.npy')
                    stain_matrix_path.unlink()
                    stain_reference_path = stain_references_dir/(slide_id + '.png')
                    stain_reference_path.unlink()
                    removed_stain_references += 1
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    print(f"(XL) Renamed {removed_prob_maps0} maps, {removed_prob_maps1} shifted maps, {removed_masks} masks, {removed_annotations} annotations and {removed_stain_references} stain references")