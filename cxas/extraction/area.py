from .func_helpers import get_area
import numpy as np
from cxas.label_mapper import id2label_dict


def get_all_areas(mask, img=None, draw = False):
    out = {}
    for i in range(mask.shape[0]):
        x = get_area(mask[i].astype(np.uint8), 1, 1)
        out[id2label_dict[str(i)]+'_area'] = x
    return out 
