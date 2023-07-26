from .compactness import get_all_compactness
from .area import get_all_areas
from .centroid import get_centroids
from .cardiothoracic_ratio import get_cardiothoracic_ratio
from .spinecenter_distance import get_spine_center_distance
from .perimeter import get_all_perimeters
from .bounding_box import get_all_bounding_boxes
from .convexity import get_all_convexities
import numpy as np

class Extractor():
    def __init__(self,):
        """
        """
        self.methods = {
            'Cardio-Thoracic Ratio': get_cardiothoracic_ratio,
            'CTR': get_cardiothoracic_ratio,
            'Spine-Center Distance': get_spine_center_distance,
            'SCD': get_spine_center_distance,
            'perimeter': get_all_perimeters,
            'compactness': get_all_compactness,
            'area': get_all_areas,
            'centroid': get_centroids,
            'box': get_all_bounding_boxes,
            'convexity': get_all_convexities,

        }
    
    def extract(self, 
                file:np.array, 
                method:str, 
                image: np.array = None,
                draw:bool=False
               ) -> dict:
        """
        """
        assert method in list(self.methods.keys()), f'Method in question ({method}) is not yet implemented. Please write an issue if you want to have it implemented.'
        return self.methods[method](file, image, draw)