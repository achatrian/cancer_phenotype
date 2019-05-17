import os
import sys
sys.path.extend(['../../..', '../../base'])
from pathlib import Path
import json
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from base.utils.annotation_builder import AnnotationBuilder
from quant import read_annotations, contour_to_mask, mark_point_on_mask


def test_merge_contours():
    annotation_path = Path(
        '/home/andrea/Documents/Repositories/AIDA/dist/data/annotations/17_A047-4463_153D+-+2017-05-11+09.40.22.json')
    annotation = AnnotationBuilder.from_annotation_path(annotation_path)
    contours, name = annotation.get_layer_points('epithelium', contour_format=True)
    centroids = np.array(annotation.get_paths_centroids_and_radii('epithelium'))
    print(len(contours), len(centroids))
    dist_matrix = cdist(centroids, centroids, 'euclidean')  # Parameters
    idx = 50
    centroid_thresh, closeness_thresh, dissimilarity_thresh = 500.0, 500.0, 500.0
    close_contour_idx = np.where(dist_matrix[idx, :] < centroid_thresh)[0]
    closest_contour_idx = np.argmin(
        dist_matrix[idx, list(range(idx)) + list(range(idx + 1, dist_matrix.shape[1]))])  # find closest contour
    close_contours = tuple(contours[idx] for idx in close_contour_idx)
    assert contours[idx] is not contours[closest_contour_idx]
    len(close_contours)
    bb0 = cv2.boundingRect(contours[idx])
    check_results = []
    closest_rect_positions = 'none'
    for contour in close_contours:
        bb1 = cv2.boundingRect(contour)
        rects_positions, origin_rect, rect_areas = AnnotationBuilder.check_relative_rect_positions(bb0, bb1, eps=10)
        check_results.append((rects_positions, origin_rect, rect_areas))
        if contour is contours[closest_contour_idx]:
            bb_closest = bb1
            closest_rect_positions = rects_positions
    # extract items and their points for processing
    pick_contour = 0
    idx1 = close_contour_idx[pick_contour]
    item0 = annotation['epithelium']['items'][idx]
    item1 = annotation['epithelium']['items'][idx1]
    assert item0 is not item1
    points_near, points_far = (tuple(annotation.item_points(item0 if origin_rect == 0 else item1)),
                               tuple(annotation.item_points(item0 if origin_rect == 1 else item1)))
    print(len(points_near), len(points_far), '| 0 or 1?', int(points_near != tuple(annotation.item_points(item0))))
    if closest_rect_positions == 'overlap':
        # if contours overlap, remove overlapping points from the bottom / rightmost contour
        points_far = annotation.remove_overlapping_points(points_near, points_far)
        print("After removing from point far len is ", len(points_far))
    # check out points on masks
    contour_near = contours[idx] if origin_rect == 0 else contours[idx1]
    contour_far = contours[idx] if origin_rect == 1 else contours[idx1]
    assert np.array_equal(np.array(points_near).astype(np.int32)[:, np.newaxis, :],
                          contour_near)  # points and contour are the same
    assert np.array_equal(np.array(points_far).astype(np.int32)[:, np.newaxis, :], contour_far)
    mask0 = contour_to_mask(contour_near).astype(np.uint8)
    mask1 = contour_to_mask(contour_far).astype(np.uint8)
    contour_rect0 = cv2.boundingRect(contour_near)
    contour_rect1 = cv2.boundingRect(contour_far)
    print('bb0', contour_rect0, 'bb1', contour_rect1)
    # closest_points, point_dist = annotation.find_closest_points(annotation.euclidean_dist,
    #                                                             points_near, points_far,
    #                                                             contour_rect0, contour_rect1,
    #                                                             check_results[pick_contour][0],
    #                                                             closeness_thresh)
    extreme_points = annotation.find_extreme_points(points_near, points_far, check_results[pick_contour][0], closeness_thresh)
    outer_points = annotation.get_merged(points_near, points_far, extreme_points)
    assert outer_points
