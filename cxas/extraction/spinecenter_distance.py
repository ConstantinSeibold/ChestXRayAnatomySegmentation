from sklearn.linear_model import LinearRegression
import numpy as np
from PIL import Image
from cxas.label_mapper import label_mapper
from cxas.extraction.func_helpers import get_centers, sort_by_distance, get_min_dist
from cxas.extraction.draw_helpers import draw_point, draw_line

def get_spine_center_distance(mask, img=None, draw = False):
    """
    Calculate Spine-Center Distance. Distance from individual vertebrae to a regressed center line from all vertebrae.
    
    Parameters
    ----------
        mask: mask in form of np array [n_classes, width, height]
        img: source image, only used for visualization
        draw: whether to visualize the features
        
    Returns
    -------
        SCD: Distance from individual vertebrae to a regressed center line from all vertebrae.
    """
    def get_reg_line(points):
        x = np.array([c[0] for c in points])
        y = np.array([c[1] for c in points]).reshape(-1,1)

        reg = LinearRegression().fit( y, x)

        x_new = np.array(range(512)).reshape(-1,1)
        y_new = np.array([int(r) for r in reg.predict(x_new)])

        cc = np.array(list(zip(y_new,x_new[:,0])))

        return cc
    
    centers = get_centers(mask, label_mapper['all vertebrae'])
    centers = [c for c in centers if (c[0]>0) and (c[1] >0)]
    centers = sort_by_distance((256, 0), centers)

    cc = get_reg_line(centers)
    cc_ = [(c[0],c[1]) for c in cc]
    
    points, center_dists = get_min_dist(centers,cc)

    points = np.array(points)
    cc_ = [c for c in cc_ if (c[1]>centers[0][1]) and (c[1]<centers[-1][1])]
    
    
    # img, cc, 
    if draw:
        width = 8

        if img is None:
            img = Image.new("RGB", (mask.shape[1], mask.shape[2]), "black")
        else:
            pass

        for idx in range(len(cc_[:-1])):
            img = draw_line(img, cc_[idx], cc_[idx+1], '#2A9D8F',width)

        for idx in range(len(centers[:-1])):
            img = draw_line(img, centers[idx], centers[idx+1], '#F4A261',width)

        for idx in range(len(centers)):
            img = draw_line(img, centers[idx], (points[idx][0],points[idx][1]), '#264653',width)

        for idx in range(len(centers)):
            img = draw_point(img, centers[idx], '#E76F51',width*2)

        img = draw_point(img, cc_[0], '#264653',width*2)
        img = draw_point(img, cc_[-1], '#264653',width*2)
        
        return {'score':np.array(center_dists).mean(), 'drawing':img}
    return {'score':np.array(center_dists).mean()}