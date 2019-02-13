import json
import math
import copy
from collections import defaultdict, OrderedDict
from itertools import combinations, product, chain
import logging
from datetime import datetime
from pathlib import Path
import tqdm
import numpy as np
from scipy.misc import comb
import cv2


class AIDAnnotation:
    """
    Use to create AIDA_annotation.json file for a single image/WSI that can be read by AIDA
    """
    def __init__(self, slide_id, project_name, layers=(), keep_original=False):
        self.slide_id = slide_id  # AIDA uses filename before extension to determine which annotation to load
        self.project_name = project_name
        self._obj = {
            'name': self.project_name,
            'layers': []
        }  # to be dumped to .json
        self.layers = []
        # store additional info on segments for processing
        self.metadata = defaultdict(lambda: {'tile_dict': [], 'dist': []})  # layer_idx -> (metadata_name -> (item_idx -> value)))
        self.last_added_item = None
        # point comparison
        self.keep_original = keep_original
        for layer_name in layers:
            self.add_layer(layer_name)  # this also updates self.layers with references to layers names

    @staticmethod
    def euclidean_dist(p1, p2):
        x1, y1 = (p1['x'], p1['y']) if isinstance(p1, dict) else p1
        x2, y2 = (p2['x'], p2['y']) if isinstance(p2, dict) else p2
        return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

    def __getitem__(self, idx):
        return self._obj['layers'][idx]

    def __setitem__(self, idx, value):
        self._obj['layers'][idx] = value

    def has_layers(self):
        return bool(self._obj['layers'])

    def layer_has_items(self, layer_idx):
        if type(layer_idx) is str:
            layer_idx = self.get_layer_idx(layer_idx)  # if name is given rather than index
        layers_len = len(self._obj['layers'][layer_idx])
        if layer_idx > layers_len:
            raise IndexError(f'Index {layer_idx} is out of range for layers list with len {layers_len}')
        layer = self._obj['layers'][layer_idx]
        return bool(layer['items'])

    def get_layer_idx(self, layer_name):
        try:
            # turn into numerical index
            layer_idx = next(i for i, layer in enumerate(self._obj['layers']) if layer['name'] == layer_name)
        except StopIteration:
            raise ValueError(f"No layer with specified name {layer_name}")
        return layer_idx

    def add_layer(self, layer_name):
        new_layer = {
            'name': layer_name,
            'opacity': 1,
            'items': []
        }
        self._obj['layers'].append(new_layer)
        self.layers.append(self._obj['layers'][-1]['name'])
        self.metadata[self.layers[-1]]['tile_rect'] = []
        return self

    def add_item(self, layer_idx, type_, class_=None, points=None, tile_rect=None):
        r"""
        Add item to desired layer
        :param layer_idx: numerical index or layer name (converted to index)
        :param type_:
        :param class_:
        :param points:
        :param tile_rect:
        :return:
        """
        if type(layer_idx) is str:
            layer_idx = self.get_layer_idx(layer_idx)  # if name is given rather than index
        new_item = {
            'class': class_ if class_ else self._obj['layers'][layer_idx]['name'],
            'type': type_,
            "segments": [],
            'closed': True
        }  # TODO properties must change with item type - e.g. circle, rectangle
        self._obj['layers'][layer_idx]['items'].append(new_item)
        self.last_added_item = self._obj['layers'][layer_idx]['items'][-1]  # use to update item
        if points:
            self.add_segments_to_last_item(points)
        if tile_rect:
            # add tile info to metadata
            if not isinstance(tile_rect, (tuple, list)) or len(tile_rect) != 4:
                raise ValueError("Invalid tile rect was passed - must be a tuple (x, y, w, h) specifying the bounding box around the item's segments")
            self.metadata[self.layers[layer_idx]]['tile_rect'].append(tile_rect)
        return self

    def remove_item(self, layer_idx, item_idx):
        layer_idx = self.get_layer_idx(layer_idx)  # FIXME slightly confusing - this is getting layer idx from name or doing nothing
        del self._obj['layers'][layer_idx]['items'][item_idx]
        layer_name = self._obj['layers'][layer_idx]['name']
        del self.metadata[layer_name]['tile_rect'][item_idx]

    def set_layer_items(self, layer_idx, items):
        self._obj['layers'][layer_idx]['items'] = items
        return self

    def add_segments_to_last_item(self, points):
        segments = []
        for i, point in enumerate(points):
            x, y = point
            new_segment = {
                'point': {
                    'x': x,
                    'y': y,
                }
              }
            segments.append(new_segment)
        self.last_added_item['segments'] = segments
        return self

    def merge_overlapping_segments(self, closeness_thresh=3.0, dissimilarity_thresh=5.0, max_iter=1):
        """
        Compares all segments and merges overlapping ones
        """
        # Set up logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'merge_segments_{datetime.now()}.log')  # logging to file for debugging
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()  # logging to console for general runtime info
        ch.setLevel(logging.DEBUG)  # if this is set to ERROR, then errors are not printed to log, and vice versa
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info("Begin segment merging ...")
        # Merge by layer first - this can removes some uncertainty as complete paths are usually better classified than interrupted ones
        for layer in self._obj['layers']:
            changed = True
            num_iters = 0
            items = layer['items']
            logger.info(f"[Layer '{layer['name']}'] Initial number of items is {len(items)}.")
            merged_num = 0  # keep count of change in elements number
            while changed and num_iters < max_iter:
                # repeat with newly formed contours
                items = layer['items']
                if num_iters == 0:
                    logger.info(f"[Layer '{layer['name']}'] First iteration ...")
                else:
                    logger.info(f"[Layer '{layer['name']}'] Iter #{num_iters} cycle merged {len(store_items) - len(items)}")
                store_items = copy.deepcopy(layer['items'])  # copy to iterate over
                processed_items_idx = set()  # store indices so that cycle is not repeated if either item has already been removed / discarded
                to_remove = set()
                for (i, item0), (j, item1) in tqdm.tqdm(combinations(enumerate(store_items), 2),
                                                            leave=False, total=comb(N=len(store_items), k=2)):
                    if i == j or i in processed_items_idx or j in processed_items_idx:
                        continue  # items are the same or items have already been processed
                    # Brute force approach, shapes are not matched perfectly, so if there are many points close to another
                    # point and distance dosesn't match the points 1-to-1 paths might be merged even though they don't
                    try:
                        tile_rect0 = self.metadata[layer['name']]['tile_rect'][i]
                        tile_rect1 = self.metadata[layer['name']]['tile_rect'][j]
                    except KeyError as keyerr:
                        raise KeyError(f"{keyerr.args[0]} - Unspecified metadata for layer {layer['name']}")
                    except IndexError:
                        raise IndexError(f"Item {i if i > j else j} is missing from layer {layer['name']} metadata")
                    rects_positions, origin_rect, rect_areas = self.check_relative_rect_positions(tile_rect0, tile_rect1)
                    if not rects_positions:
                        continue  # do nothing if items are not from touching/overlapping tiles
                    points_near, points_far = (tuple(self.item_points(item0 if origin_rect == 0 else item1)),
                                               tuple(self.item_points(item0 if origin_rect == 1 else item1)))
                    if rects_positions == 'overlap':
                        points_far = self.remove_overlapping_points(points_near, points_far)
                        try:
                            points_far[0]
                        except IndexError as err:
                            logging.error(str(err.args), exc_info=True)
                    assert points_far, "Some points must remain from this operation, or positions should have been 'contained'"
                    total_min_dist = 0.0
                    # noinspection PyBroadException
                    try:
                        closest_points, point_dist = self.find_closest_points(self.euclidean_dist, points_near, points_far, closeness_thresh)
                        if closest_points and len(closest_points) > 1:
                            total_min_dist += sum(
                                min(point_dist[p0][p1] if (p0, p1) in closest_points else 0 for p1 in points_far)
                                for p0 in points_near)  # sum dist
                            if total_min_dist / len(closest_points) < dissimilarity_thresh:
                                outer_points = self.get_merged(points_near, points_far, closest_points)
                                # make boundin
                                x_out, y_out = (min(tile_rect0[0], tile_rect1[0]), min(tile_rect0[1], tile_rect1[1]))
                                w_out = max(tile_rect0[0] + tile_rect0[2], tile_rect1[0] + tile_rect1[2]) - x_out  # max(x+w) - min(x)
                                h_out = max(tile_rect0[1] + tile_rect0[3], tile_rect1[1] + tile_rect1[3]) - y_out  # max(x+w) - min(x)
                                self.add_item(layer['name'], item0['type'], class_=item0['class'], points=outer_points,
                                              tile_rect=(x_out, y_out, w_out, h_out))
                                logger.debug(f"Item {i} and {j} were merged - total dist per close point: {total_min_dist / len(closest_points)} (threshold = {dissimilarity_thresh})")
                                processed_items_idx.add(i)
                                processed_items_idx.add(j)
                                to_remove.add(i)
                                to_remove.add(j)
                    except Exception:
                        logger.debug(f"""
                        [Layer '{layer['name']}'] iter: #{num_iters} items: {(i, j)} total item num: {len(items)} merged items:{len(store_items) - len(items)}
                        [Bounding box] 0 x: {tile_rect0[0]} y: {tile_rect0[1]} w: {tile_rect0[2]} h: {tile_rect0[2]}
                        [Bounding box] 1 x: {tile_rect1[0]} y: {tile_rect1[1]} w: {tile_rect1[2]} h: {tile_rect1[2]}
                        Result of rectangle position check: '{rects_positions}', origin: {origin_rect}, areas: {rect_areas}
                        """)
                        logger.error('Failed.', exc_info=True)
                for item_idx in sorted(to_remove, reverse=True):  # remove items that were merged - must start from last index or will return error
                    # noinspection PyBroadException
                    try:
                        self.remove_item(layer['name'], item_idx)
                    except Exception:
                        logger.error(f"Failed to remove item {item_idx} in layer {layer['name']}", exc_info=True)
                changed = items != store_items
                merged_num += len(store_items) - len(items)
                num_iters += 1
            if changed:
                logger.info(f"[Layer '{layer['name']}'] Max number of iterations reached ({num_iters})")
            else:
                logger.info(f"[Layer '{layer['name']}'] No changes after {num_iters} iterations.")
            logger.info(f"[Layer '{layer['name']}'] {merged_num} total merged items.")

    @staticmethod
    def find_closest_points(distance, points_near, points_far, closeness_thresh=3.0):
        """
        :param distance: distance metric taking two points and returning commutative value
        :param points_near:
        :param points_far:
        :param closeness_thresh:
        :return: pairs of corresponding points
                 distance between point pairs
        """
        point_dist = dict()  # point1 -> point2
        for (k, p0), (l, p1) in product(enumerate(points_near), enumerate(points_far)):
            # distance comparison
            try:
                point_dist[p0][p1] = distance(p0, p1)  # store distances
            except KeyError:
                point_dist[p0] = OrderedDict()
                point_dist[p0][p1] = distance(p0, p1)  # store distances
            if tuple(point_dist[p0].keys()) != points_far[0:l + 1]:  # must have same order as well as same elements
                assert tuple(point_dist[p0].keys()) == points_far[0:l + 1]  # must have same order as well as same elements
        closest_points = set()
        for p0 in points_near:
            closest_idx = np.argmin(list(point_dist[p0].values())).item()  # must list() value_view, as it is not a sequence
            closest_point = points_far[closest_idx]
            if point_dist[p0][closest_point] < closeness_thresh:
                closest_points.add((p0, points_far[closest_idx]))
        return closest_points, point_dist

    @staticmethod
    def get_merged(points_near, points_far, close_point_pairs, positions='horizontal'):
        """
        Function to merge contours from adjacent tiles
        :param points_near: leftmost (for positions = horizontal) or topmost (for position = vertical) path
        :param points_far:
        :param close_point_pairs:
        :param positions
        :return: merged points
        """
        close_point_pairs = tuple(close_point_pairs)
        # two-way mapping
        correspondance = {p0: p1 for p0, p1 in close_point_pairs}
        correspondance.update({p1: p0 for p0, p1 in close_point_pairs})
        close_points_near = set(p0 for p0, p1 in close_point_pairs)
        close_points_far = set(p1 for p0, p1 in close_point_pairs)
        # drop close segments
        outer_points = []
        if positions == '':
            raise ValueError("Cannot merge items if bounding boxes are not at least adjacent")
        # https://stackoverflow.com/questions/45323590/do-contours-returned-by-cvfindcontours-have-a-consistent-orientation
        # outer contours are oriented counter-clockwise
        # scanning for first point is done from top left to bottom right
        # assuming contours are extracted for each value independently
        for i0, point0 in enumerate(points_near):
            # start - lower extreme - higher - extreme
            outer_points.append(point0)
            if point0 in close_points_near and i0 != 0:  # p0 of lowest (p0, p1) pair
                break
        assert point0 in close_points_near, "This loop should end at a close point"
        start_p1 = correspondance[point0]
        start_p1_idx = points_far.index(start_p1)
        for i1, point1 in enumerate(points_far[start_p1_idx:] + points_far[:start_p1_idx]):
            outer_points.append(point1)
            if point1 in close_points_far and i1 != 0:  # p1 of highest (p0, p1) pair
                break
        assert point1 in close_points_far, "This loop should end at a close point"
        restart_p0 = correspondance[point1]
        for i00, point0 in enumerate(points_near[points_near.index(restart_p0):]):
            outer_points.append(point0)
            i0 += 1
        return outer_points

    @staticmethod
    def remove_overlapping_points(points_near, points_far):
        points_far = list(points_far)
        contour_near = np.array(points_near)[:, np.newaxis, :]
        for i1, point1 in reversed(list(enumerate(copy.copy(points_far)))):
            # cv2.pointPolygonTest
            # Positive value if the point is inside the contour !!!
            # Negative value if the point is outside the contour
            # Zero if the point is on the contour
            if cv2.pointPolygonTest(contour_near, point1, False) >= 0:
                del points_far[i1]
        return tuple(points_far)

    @staticmethod
    def item_points(item):
        """Generator over item's points"""
        # remove multiple identical points - as they can cause errors
        points = set()
        for segment in item['segments']:
            point = tuple(segment['point'].values())
            if point not in points:
                points.add(point)
                yield point

    @staticmethod
    def check_relative_rect_positions(tile_rect0, tile_rect1, eps=0):
        """
        :param tile_rect0:
        :param tile_rect1:
        :param eps: tolerance in checks
        :return: positions: overlap|horizontal|vertical|'' - the relative location of the two paths
                 origin_rect: meaning depends on relative positions of two boxes:
                            * contained: which box is bigger
                            * horizontal: leftmost box
                            * vertical: topmost box
                            * overlap: topmost box
                            * '': None
                 rect_areas: are of the bounding boxes
        """
        x0, y0, w0, h0 = tile_rect0
        x1, y1, w1, h1 = tile_rect1
        x_w0, y_h0, x_w1, y_h1 = x0 + w0, y0 + h0, x1 + w1, y1 + h1
        # symmetric relationship - check only in one
        x_overlap = (x0 - eps <= x1 <= x_w0 + eps or x0 - eps <= x_w1 <= x_w0 + eps)  # no need for symmetric check
        y_overlap = (y0 - eps <= y1 <= y_h0 + eps or y0 - eps <= y_h1 <= y_h0 + eps)
        x_contained = (x0 - eps <= x1 <= x_w0 + eps and x0 - eps <= x_w1 <= x_w0 + eps) or \
                      (x1 - eps <= x0 <= x_w1 + eps and x1 - eps <= x_w0 <= x_w1 + eps)  # one is bigger than the other - not symmetric!
        y_contained = (y0 - eps <= y1 <= y_h0 + eps and y0 - eps <= y_h1 <= y_h0) or \
                      (y1 - eps <= y0 <= y_h1 + eps and y1 - eps <= y_h0 <= y_h1)
        if x_contained and y_contained:
            positions = 'contained'
            if (x_w0 - x0) * (y_h0 - y0) >= (x_w1 - x1) * (y_h1 - y1):
                origin_rect = 0  # which box is bigger
            else:
                origin_rect = 1
        elif not x_contained and y_overlap and (x_w0 < x1 + eps or x_w1 < x0 + eps):
            positions = 'horizontal'
            if x_w0 < x1 + eps:
                origin_rect = 0
            elif x_w1 < x0 + eps:
                origin_rect = 1
            else:
                raise ValueError("shouldn't be here")
        elif not y_contained and x_overlap and (y_h0 < y1 + eps or y_h1 < y0 + eps):
            positions = 'vertical'
            if y_h0 < y1 + eps:
                origin_rect = 0
            elif y_h1 < y0 + eps:
                origin_rect = 1
            else:
                raise ValueError("shouldn't be here")
        elif x_overlap and y_overlap:
            positions = 'overlap'
            if y0 <= y1:
                origin_rect = 0
            else:
                origin_rect = 1
        else:
            positions = ''
            origin_rect = None
        rect_areas = ((x_w0 - x0) * (y_h0 - y0), (x_w1 - x1) * (y_h1 - y1))
        return positions, origin_rect, rect_areas,

    def print(self, indent=4):
        print(json.dumps(self._obj, sort_keys=False, indent=indent))

    def dump_to_json(self, save_dir, suffix_to_remove=('.ndpi', '.svs')):
        save_path = Path(save_dir)/self.slide_id
        save_path = save_path.with_suffix('.json') if save_path.suffix in suffix_to_remove else \
            save_path.parent/(save_path.name+'.json')  # add json taking care of file ending in .some_text.[ext,no_ext]
        json.dump(self._obj, open(save_path, 'w'))

    def get_layer_points(self, layer_idx, contour_format=False):
        if isinstance(layer_idx, str):
            layer_idx = self.get_layer_idx(layer_idx)
        layer = self._obj['layers'][layer_idx]
        if contour_format:
            layer_points = list(np.array(list(self.item_points(item)))[:, np.newaxis, :] for item in layer['items'])
        else:
            layer_points = list(list(self.item_points(item)) for item in layer['items'])
        return layer_points, layer['name']



# @staticmethod
# def find_extremes(close_point_pairs, distance):
#     """
#     Finds extreme points based on close point pairs between two shapes
#     :param close_point_pairs:
#     :param distance:
#     :return:
#     """
#     close_point_pairs = tuple(close_point_pairs)
#     # find vector representation of line
#     points = tuple(p for p in chain.from_iterable(close_point_pairs))
#     x = np.array(tuple([1, p[0]] for p in points))  # first basis function is for bias, second is linear (slope)
#     y = np.array(tuple(p[1] for p in points))
#     weights = np.linalg.pinv(x.T @ x) @ x.T @ y
#     bias, slope = weights
#     offset = np.array([0, slope*0 + bias])
#     direction = np.array([1, slope*1 + bias])
#     def find_projection(v, r, s):
#         return r + s * np.dot(v, s) / np.dot(s, s)
#     projection_pairs = [(find_projection(p0, offset, direction), find_projection(p1, offset, direction))
#                         for p0, p1 in close_point_pairs]
#     centre_of_mass = np.mean([p for p in chain.from_iterable(projection_pairs)])  # points are aligned
#     extremes = [
#         close_point_pairs[
#             np.argmin([min(distance(p0, centre_of_mass), distance(p1, centre_of_mass))
#                        for p0, p1 in projection_pairs])
#         ],  # highest points
#         close_point_pairs[
#             np.argmax([max(distance(p0, centre_of_mass), distance(p1, centre_of_mass))
#                        for p0, p1 in projection_pairs])
#         ]  # lowest points
#     ]
#     topmost_first_extremes = sorted(extremes, key=lambda point_pair: point_pair[0][1])  # y-axis of first point
#     return topmost_first_extremes





#
# def get_merged(points_near, points_far, close_point_pairs, positions='horizontal'):
#     """
#     Function to merge contours from adjacent tiles
#     :param points_near: leftmost (for positions = horizontal) or topmost (for position = vertical) path
#     :param points_far:
#     :param close_point_pairs:
#     :param positions
#     :return: merged points
#     """
#     close_point_pairs = tuple(close_point_pairs)
#     # two-way mapping
#     correspondance = {p0: p1 for p0, p1 in close_point_pairs}
#     correspondance.update({p1: p0 for p0, p1 in close_point_pairs})
#     # drop close segments
#     outer_points = []
#     if positions == '':
#         raise ValueError("Cannot merge items if bounding boxes are not at least adjacent")
#     if positions == 'overlap':
#         raise NotImplementedError(f"No need to merge overlapping contours")
#     # https://stackoverflow.com/questions/45323590/do-contours-returned-by-cvfindcontours-have-a-consistent-orientation
#     # outer contours are oriented counter-clockwise
#     # scanning for first point is done from top left to bottom right
#     # assuming contours are extracted for each value independently
#     if positions == 'horizontal':
#         extremes = [
#             close_point_pairs[
#                 np.argmin([min(p0[1], p1[1]) for p0, p1 in close_point_pairs])
#             ],  # highest points
#             close_point_pairs[
#                 np.argmax([max(p0[1], p1[1]) for p0, p1 in close_point_pairs])
#             ]  # lowest points
#         ]
#         for i0, point0 in enumerate(points_near):
#             # start - lower extreme - higher - extreme
#             outer_points.append(point0)
#             if point0 == extremes[1][0]:  # p0 of lowest (p0, p1) pair
#                 break
#         start_p1 = correspondance[point0]
#         start_p1_idx = points_far.index(start_p1)
#         for i1, point1 in enumerate(points_far[start_p1_idx:] + points_far[:start_p1_idx]):
#             outer_points.append(point1)
#             if point1 == extremes[0][1]:  # p1 of highest (p0, p1) pair
#                 break
#         restart_p0 = correspondance[point1]
#         for i00, point0 in enumerate(points_near[points_near.index(restart_p0):]):
#             outer_points.append(point0)
#             i0 += 1
#         assert outer_points
#     elif positions == 'vertical' or positions == 'overlap':
#         extremes = [
#             close_point_pairs[
#                 np.argmin([min(p0[0], p1[0]) for p0, p1 in close_point_pairs])
#             ],  # leftmost points
#             close_point_pairs[
#                 np.argmax([max(p0[0], p1[0]) for p0, p1 in close_point_pairs])
#             ]  # rightmost points
#         ]
#         for i0, point0 in enumerate(points_near):
#             # start - lower extreme - higher - extreme
#             outer_points.append(point0)
#             if point0 == extremes[0][0]:  # p0 of rightmost (p0, p1) pair
#                 break
#             assert i0 < extremes[0] and i0 < extremes[1]
#         start_p1 = correspondance[point0]
#         start_p1_idx = points_far.index(start_p1)
#         for i1, point1 in enumerate(points_far[start_p1_idx:] + points_far[:start_p1_idx]):
#             outer_points.append(point1)
#             if point1 == extremes[1][1]:  # p1 of rightmost (p0, p1) pair
#                 break
#         restart_p0 = correspondance[point1]
#         for i00, point0 in enumerate(points_near[points_near.index(restart_p0):]):
#             outer_points.append(point0)
#             i0 += 1
#         assert outer_points
#     return outer_points