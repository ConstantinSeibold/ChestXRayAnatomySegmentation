# https://github.com/chrise96/image-to-coco-json-converter/tree/master

from PIL import Image
import numpy as np
from skimage import measure
import os
import json


def create_category_annotation(category_dict):
    """
    Create category annotations in COCO JSON format.

    Args:
        category_dict (dict): Dictionary containing category names and IDs.

    Returns:
        list: List of category annotations.
    """
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def get_coco_json_format():
    """
    Get the standard COCO JSON format.

    Returns:
        dict: COCO JSON format skeleton.
    """
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format
