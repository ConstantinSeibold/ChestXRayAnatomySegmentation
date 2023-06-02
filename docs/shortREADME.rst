================================
Chest X-Ray Anatomy Segmentation
================================

--------------
Installation
--------------

The project is available in PyPI. To install run:

``pip install cxas``


--------------------------------------
Running Segmentation from terminal
--------------------------------------

Segment the anatomy of X-Ray images \(.jpg,.png,.dcm\) and store the results \(npy,json,jpg,png,dicom-seg\):

```
cxas_segment -i {desired input directory or file} -o {desired output directory}
```

------------------------------------------
Running Feature Extraction from terminal
------------------------------------------

Extract anatomical features from X-Ray images \(.jpg,.png,.dcm\) and store the results \(.csv\):

``cxas_feat_extract -i {desired input directory or file} -o {desired output directory} -f {desired features to extract}``

----------------------------
Running either from terminal
----------------------------

Extract anatomical features from X-Ray images \(.jpg,.png,.dcm\) and store the results \(.csv\):

``cxas -i {desired input directory or file} -o {desired output directory} -mode {"segment" or "exract"} -f {required if mode == 'extract'}``



--------------
Citation
--------------

If you use this work or dataset, please cite:

.. code:: bibtex

    @inproceedings{Seibold_2022_BMVC,
        author    = {Constantin Marc Seibold and Simon Reiß and M. Saquib Sarfraz and Matthias A. Fink and Victoria Mayer and Jan Sellner and Moon Sung Kim and Klaus H. Maier-Hein and Jens Kleesiek and Rainer Stiefelhagen},
        title     = {Detailed Annotations of Chest X-Rays via CT Projection for Report Understanding},
        booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
        publisher = {{BMVA} Press},
        year      = {2022},
        url       = {https://bmvc2022.mpi-inf.mpg.de/0058.pdf}
    }
    
.. code:: bibtex

    @inproceedings{Seibold_2023_CXAS,
        author    = {Constantin Seibold, Alexander Jaus, Matthias Fink,
        Moon Kim, Simon Reiß, Jens Kleesiek*, Rainer Stiefelhagen*},
        title     = {Accurate Fine-Grained Segmentation of Human Anatomy in Radiographs via Volumetric Pseudo-Labeling},
        year      = {2023},
    }
