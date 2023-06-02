import numpy as np
import math

def get_center_of_mask(mask):
    y_coords, x_coords = np.nonzero(mask)
    center = (np.mean(x_coords).astype(np.int32), np.mean(y_coords).astype(np.int32))
    return center

def get_centers(mask, ids):
    centers = [get_center_of_mask(mask[id][0]) for id in ids if mask[id][0].sum()>0]
    return centers


def sort_by_distance(reference_point, coordinates):
    distances = []
    for point in coordinates:
        x1, y1 = reference_point
        x2, y2 = point
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append((point, dist))
    distances.sort(key=lambda x: x[1])
    return [point for (point, dist) in distances]

def get_min_dist(points1, points2):
    out_points = []
    out_dists = []
    # import pdb; pdb.set_trace()
    for point in points1:
        differences = points2 - point
        distances = np.linalg.norm(differences, axis=1)
        min_index = np.argmin(distances)
        out_points += [points2[min_index]]
        out_dists += [distances[min_index]]
    # import pdb; pdb.set_trace()
    return out_points, out_dists