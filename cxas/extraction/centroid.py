import numpy as np
import cv2
from cxas.label_mapper import id2label_dict


def get_centroids(mask, img=None, draw = False):
    out = {}
    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            x,y = -1, -1
        else:
            x,y = get_centroid(mask[i].astype(np.uint8), 1, 1)
        out[id2label_dict[str(i)]+'_cx'] = x
        out[id2label_dict[str(i)]+'_cy'] = y
    return out 

def get_centroid(mask, spacing_x, spacing_y):
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    centroid = (cx, cy)
    return np.array(centroid)