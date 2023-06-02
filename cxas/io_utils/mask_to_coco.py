import numpy as np
import json
from pycocotools import mask

# Function to convert binary mask to COCO RLE format
def binary_mask_to_rle(
                        binary_mask:np.array,
                    ):
    """
    """
    mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    mask_encoded['counts'] = mask_encoded['counts'].decode('utf-8')
    return mask_encoded

def mask_to_annotation(
                        mask:np.array, 
                        base_ann_id:int=1,
                        img_id:int = 1,
                        ):
    """
    """
    annotations = []
    for i in range(mask.shape[0]):
        if mask[i].sum()==0:
            continue
        binary_mask = mask[i]
        # import pdb;pdb.set_trace()
        mask_encoded = binary_mask_to_rle(binary_mask)
        annotation = {
            'id': base_ann_id + i,
            'image_id': img_id,  # Assuming all masks belong to the same image
            'category_id': i,  # Assuming a single category
            'segmentation': mask_encoded,
            'area': int(np.sum(binary_mask)),
            'bbox': toBox(mask_encoded).tolist(),
            'iscrowd': 0  # Set to 1 if the mask represents a crowd region
        }
        annotations.append(annotation)
    return annotations


def toBox(binary_mask):
    return mask.toBbox(binary_mask)