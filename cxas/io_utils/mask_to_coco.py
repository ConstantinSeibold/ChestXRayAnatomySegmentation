import numpy as np
from pycocotools import mask
from copy import deepcopy

def binary_mask_to_rle(binary_mask: np.array) -> dict:
    """
    Convert binary mask to COCO RLE format.

    Args:
        binary_mask (np.array): Binary mask array.

    Returns:
        dict: COCO RLE encoded mask.
    """
    mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    mask_encoded['counts'] = mask_encoded['counts'].decode('utf-8')
    return mask_encoded
                      
def rle_to_binary_mask(rle_mask: dict) -> np.array:
    """
    Convert COCO RLE encoded mask to binary mask.

    Args:
        rle_mask (dict): COCO RLE encoded mask.

    Returns:
        np.array: Binary mask array.
    """
    rle_mask_copy = deepcopy(rle_mask)
    rle_mask_copy['counts'] = rle_mask_copy['counts'].encode('utf-8')
    binary_mask = mask.decode(rle_mask_copy)
    return binary_mask
  
def mask_to_annotation(mask: np.array, base_ann_id: int = 1, img_id: int = 1) -> list:
    """
    Convert mask array to COCO annotation format.

    Args:
        mask (np.array): Mask array.
        base_ann_id (int, optional): Base annotation ID. Defaults to 1.
        img_id (int, optional): Image ID. Defaults to 1.

    Returns:
        list: List of COCO annotations.
    """
    annotations = []
    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            continue
        binary_mask = mask[i]
        mask_encoded = binary_mask_to_rle(binary_mask)
        annotation = {
            'id': base_ann_id + i,
            'image_id': img_id,
            'category_id': i,
            'segmentation': mask_encoded,
            'area': int(np.sum(binary_mask)),
            'bbox': toBox(mask_encoded).tolist(),
            'iscrowd': 0  # Set to 1 if the mask represents a crowd region
        }
        annotations.append(annotation)
    return annotations

def toBox(binary_mask: np.array) -> list:
    """
    Convert binary mask to bounding box coordinates.

    Args:
        binary_mask (np.array): Binary mask array.

    Returns:
        list: Bounding box coordinates.
    """
    return mask.toBbox(binary_mask)
