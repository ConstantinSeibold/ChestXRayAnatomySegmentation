import numpy as np
from pycocotools import mask
from copy import deepcopy

def binary_mask_to_rle(binary_mask: np.array) -> dict:
    """
    Convert a binary mask to COCO Run Length Encoding (RLE) format.
    
    Args:
        binary_mask (np.array): Binary mask array.

    Returns:
        dict: Dictionary containing the RLE-encoded mask.
    """
    mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    mask_encoded['counts'] = mask_encoded['counts'].decode('utf-8')
    return mask_encoded

def rle_to_binary_mask(rle_mask: dict) -> np.array:
    """
    Convert a RLE-encoded mask to a binary mask array.

    Args:
        rle_mask (dict): Dictionary containing the RLE-encoded mask.

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
        base_ann_id (int): Base annotation ID.
        img_id (int): Image ID.

    Returns:
        list: List of annotations in COCO format.
    """
    annotations = []
    for i in range(mask.shape[0]):
        if mask[i].sum() == 0:
            continue
        binary_mask = mask[i]
        mask_encoded = binary_mask_to_rle(binary_mask)
        annotation = {
            'id': base_ann_id + i,
            'image_id': img_id,  # Assuming all masks belong to the same image
            'category_id': i,  # Assuming a single category
            'segmentation': mask_encoded,
            'area': int(np.sum(binary_mask)),
            'bbox': to_box(mask_encoded).tolist(),
            'iscrowd': 0  # Set to 1 if the mask represents a crowd region
        }
        annotations.append(annotation)
    return annotations

def to_box(binary_mask: dict) -> list:
    """
    Convert binary mask to bounding box format.

    Args:
        binary_mask (dict): Binary mask.

    Returns:
        list: Bounding box coordinates.
    """
    return mask.toBbox(binary_mask)
