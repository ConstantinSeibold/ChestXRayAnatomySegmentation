from .cardiothoracic_ratio import get_cardiothoracic_ratio
from .spinecenter_distance import get_spine_center_distance
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