import numpy as np
from PIL import Image
from cxas.label_mapper import label_mapper
from cxas.extraction.draw_helpers import draw_point, draw_line

def get_cardiothoracic_ratio(npy, img=None, draw = False):
    """
    Calculate Cardio-thoracic-ratio from mask (check https://en.wikipedia.org/wiki/Cardiomegaly)
    MRD = greatest perpendicular diameter from midline to right heart border
    MLD = greatest perpendicular diameter from midline to left heart border
    ID = internal diameter of chest at level of right hemidiaphragm

    Parameters
    ----------
        npy: mask in form of np array [n_classes, width, height]
        img: source image, only used for visualization
        draw: whether to visualize the features
        
    Returns
    -------
        CTR: (MRD + MLD) / ID
    """

    def get_longest(mask):
        min_ = 0
        id_ = 0
        points = (0,0),(0,0)
        for i in range(len(mask)):
            if mask[i].sum() == 0:
                continue
            nz = mask[i].nonzero()[0]
            if len(nz)!=0:
                dist = nz.max() - nz.min()
                if dist>min_:
                    min_ = dist
                    id_ = i
                    points = [(i,nz.min()), (i,nz.max())]
        return min_, id_, points
    
    def distance_to_midline(midline, heart):
        mrd = 0
        mld = 0
        mrd_id = 0
        mld_id = 0
        min_pos, max_pos = (0,0),(0,0)
        midline1, midline2 = (0,0), (0,0)
        for i in range(len(heart)):
            if heart[i].sum() == 0:
                continue
            nz = heart[i].nonzero()[0]
            if len(nz)!=0:
                dist_ld = nz.max() - midline
                dist_rd = midline - nz.min()

                if dist_rd > mrd:
                    mrd = dist_rd
                    mrd_id = i
                    min_pos = (i,nz.min())
                    midline1 = (i, midline)
                if dist_ld > mld:
                    mld = dist_ld
                    mld_id = i
                    max_pos = (i,nz.max())
                    midline2 = (i, midline)
        return mrd, mld, mrd_id, mld_id, min_pos, max_pos, midline1, midline2
            
    if (npy[label_mapper['lung'][0]].sum()==0) or\
        (npy[label_mapper['right hemidiaphragm'][0]].sum()==0) or\
        (npy[label_mapper['heart'][0]].sum()==0):
        return {'score': -1, 'drawing': Image.new("RGB", (npy.shape[1], npy.shape[2]), "black")}
    
    lung = npy[label_mapper['lung'][0]]
    midline = npy[label_mapper['right hemidiaphragm'][0]].nonzero()[1].max()
    heart = npy[label_mapper['heart'][0]]
    
    mrd, mld, _, _, min_pos, max_pos, midline1, midline2 = distance_to_midline(midline, heart)
    ID, _, points = get_longest(lung)
    
    
    
    
    if draw:
        if img is None:
            img = Image.new("RGB", (npy.shape[1], npy.shape[2]), "black")
        else:
            pass
        
        width = 8

        img = draw_line(img, (points[0][1],points[0][0]), (points[1][1],points[1][0]), '#2A9D8F',width)
        img = draw_point(img, (points[0][1],points[0][0]), '#264653', width*2)
        img = draw_point(img, (points[1][1],points[1][0]), '#264653', width*2)


        img = draw_line(img, (min_pos[1],min_pos[0]), (midline1[1],midline1[0]), '#F4A261',width)
        img = draw_line(img, (midline2[1],midline2[0]), (midline1[1],midline1[0]), '#F4A261',width)
        img = draw_line(img, (max_pos[1],max_pos[0]), (midline2[1],midline2[0]), '#F4A261',width)

        img = draw_point(img, (min_pos[1],min_pos[0]), '#E76F51', width*2)
        img = draw_point(img, (max_pos[1],max_pos[0]), '#E76F51', width*2)
        img = draw_point(img, (midline1[1],midline1[0]), '#E76F51', width*2)
        img = draw_point(img, (midline2[1],midline2[0]), '#E76F51', width*2)
        
        return {'score':(mrd+mld)/ID, 'drawing':img}
    return {'score':(mrd+mld)/ID}