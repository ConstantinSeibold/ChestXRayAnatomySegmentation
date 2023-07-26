import cv2
import numpy as np
from cxas.label_mapper import id2label_dict


def get_all_convexities(mask, img=None, draw = False):
    out = {}
    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            convexity = -1
        else:
            convexity = get_convexity(mask[i].astype(np.uint8), 1,1).item()
        out[id2label_dict[str(i)]+'_convexity'] = convexity
    return out 

def get_convexity(mask, spacing_x, spacing_y):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = cv2.countNonZero(mask)
    
    hull = cv2.convexHull(contours[0])
    
    hull_area = cv2.contourArea(hull)
    # print('hull',area, hull_area)
    if hull_area == 0:
        return np.array(-1)
    convexity = area / hull_area
    return np.array(convexity)