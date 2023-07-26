import cv2
import numpy as np
from cxas.label_mapper import id2label_dict


def get_all_bounding_boxes(mask, img=None, draw = False):
    out = {}
    for i in range(mask.shape[0]):
        x,y,w,h = get_bounding_box(mask[i].astype(np.uint8))
        out[id2label_dict[str(i)]+'_x'] = x
        out[id2label_dict[str(i)]+'_y'] = y
        out[id2label_dict[str(i)]+'_height'] = h
        out[id2label_dict[str(i)]+'_width'] = w
    return out 

def get_bounding_box(mask, ):
    x, y, w, h = cv2.boundingRect(mask)
    bounding_box = (x, y, w, h)
    return np.array(bounding_box)