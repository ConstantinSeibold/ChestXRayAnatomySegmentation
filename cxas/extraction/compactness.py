import cv2
import numpy as np
import math
from cxas.extraction.func_helpers import get_perimeter_from_contour, get_area
from cxas.label_mapper import id2label_dict


def get_all_compactness(mask, img=None, draw = False):
    out = {}
    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            compactness = -1
        else:
            compactness = get_compactness(mask[i].astype(np.uint8), 1, 1).item()
        out[id2label_dict[str(i)]+'_compactness'] = compactness
    return out 

def get_compactness(mask, spacing_x, spacing_y):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = get_perimeter_from_contour(contours[0], spacing_x)
    area = get_area(mask, spacing_x, spacing_y) 
    # print('perimeter',area, perimeter)
    if perimeter != 0:
        compactness = 4 * math.pi * area / (perimeter * perimeter)
    else:
        return np.array(-1)
    return np.array(compactness)