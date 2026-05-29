from .func_helpers import get_area
import numpy as np
from cxas.label_mapper import id2label_dict

import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
import os
import torch.nn.functional as F
from tqdm import tqdm
import logging
# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)

def get_radiomics(mask, img=None, draw=False):
    
    image_sitk = sitk.GetImageFromArray(img)
    
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableAllImageTypes()

    out = {}
    for class_idx in tqdm(range(mask.shape[0])):
        mask_channel = mask[class_idx:class_idx+1]

        if not mask_channel.sum():
            continue

        try:
            mask_sitk = sitk.GetImageFromArray(mask_channel.astype(np.uint8))

            result = extractor.execute(image_sitk, mask_sitk, label=1)

            for key, value in result.items():
                out_key = id2label_dict[str(class_idx)] + "_" + key
                if isinstance(value, np.ndarray):
                    value = value.astype(float)
                    if value.size == 1:
                        out[out_key] = value.item()  # single float
                    else:
                        list_value = value.tolist()
                        # If list with only one element, take the element
                        out[out_key] = list_value if len(list_value) > 1 else list_value[0]
                else:
                    out[out_key] = value
        except Exception as e:
            logger.error(f"radiomics extraction failed for class {class_idx}: {e}")

    return out