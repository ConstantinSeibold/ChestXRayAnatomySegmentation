from sklearn.linear_model import LinearRegression
import numpy as np
from PIL import Image
from cxas.label_mapper import label_mapper
from cxas.extraction.func_helpers import get_perimeter_from_contour
from cxas.extraction.draw_helpers import draw_point, draw_line
import cv2
from cxas.label_mapper import id2label_dict


def get_all_perimeters(mask, img=None, draw = False):
    out = {}
    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            perimeter = -1
        else:
            perimeter = get_indv_perimeter(mask[i].astype(np.uint8), 1, 1)
        out[id2label_dict[str(i)]+'_perimeter'] = perimeter
    return out 

def get_indv_perimeter(mask, spacing_x=1, spacing_y=1):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # perimeter = cv2.arcLength(contours[0], True) * spacing_x
    perimeter = get_perimeter_from_contour(contours[0], spacing_x)
    if perimeter == 0:
        return np.array(-1)
    return perimeter
