# Chest X-Ray Anatomy Segmentation
![Title Image](./images/CXAS_logo.png)

This repository provides a way to generate fine-grained segmentations and extract understandable features of Chest X-Rays. 
Models were trained using [Multi-Label Segmentation]().

We provide demos with gradio for Chest X-Ray [**anatomy segmentation**]() and [**feature extraction**]().

![Overview](./images/Label_Overview.png)


## Installation

The project is available in PyPI. To install run:

```
pip install cxas
```

## Usage

We provide sample python code in the following notebooks:

- ![Processing different data types](demos/BasicUsage.ipynb)
- ![Processing folders of images](demos/ProcessDirectories.ipynb)
- ![Extracting features](demos/ExtractFeatures.ipynb)
- ![Visualizing Segmentations](demos/VisualizingResults.ipynb)

### Running Segmentation from terminal

Segment the anatomy of X-Ray images \(.jpg,.png,.dcm\) and store the results \(npy,json,jpg,png,dicom-seg\):

```
cxas_segment -i {desired input directory or file} -o {desired output directory}
```

<details>
<summary>Setting options</summary>
    - "-i"/"--input" : Either path to file or to directory to be processed. [**required**]
    - "-o"/"--output": Output directory for segmentation masks  [**required**]
    - "-ot"/"--output_type": Designates the storage type of segmentations if they are stored. [default = 'png']
                              choices=["json", "npy", "npz", "jpg", "png", "dicom-seg"]
    - "-g"/"--gpus": Select specific GPU/CPU to process the input. [default = "0"]
    - "-m"/"--model": Select Model used for inference. [default="UNet_ResNet50_default"]
                      choices=["UNet_ResNet50_default"]    
</details>

### Running Feature Extraction from terminal

Extract anatomical features from X-Ray images \(.jpg,.png,.dcm\) and store the results \(.csv\):

```
cxas_feat_extract -i {desired input directory or file} -o {desired output directory} -f {desired storage data format}
```

<details>
<summary>Setting options</summary>
    - "-i"/"--input" : Either path to file or to directory to be processed. [**required**]
    - "-o"/"--output": Output directory for segmentation masks  [**required**]
    - "-f", "--feature": Select which features are supposed to be extracted. [**required**]
                         choices = ["SCD", "CTR", "Spine-Center Distance","Cardio-Thoracic Ratio"]
    - "-ot"/"--output_type": Designates the storage type of segmentations if they are stored. [default = 'png']
                              choices=["json", "npy", "npz", "jpg", "png", "dicom-seg"]
    - "-g"/"--gpus": Select specific GPU/CPU to process the input. [default = "0"]
    - "-m"/"--model": Select Model used for inference. [default="UNet_ResNet50_default"]
                      choices=["UNet_ResNet50_default"]     
    - "-s"/"--store_seg": "Wether to also store segmentation masks" [default = False]    
</details>

### Running either from terminal

Extract anatomical features from X-Ray images \(.jpg,.png,.dcm\) and store the results \(.csv\):

```
cxas_feat_extract -i {desired input directory or file} -o {desired output directory} -f {desired storage data format}
```

<details>
<summary>Setting options</summary>
    - "-i"/"--input" : Either path to file or to directory to be processed. [**required**]
    - "-o"/"--output": Output directory for segmentation masks  [**required**]
    - "--mode": Select whether to segment images or extract features. [default="segment"]
                choices=["segment", 'extract']
    - "-f", "--feature": Select which features are supposed to be extracted.
                         choices = ["SCD", "CTR", "Spine-Center Distance","Cardio-Thoracic Ratio"]
    - "-ot"/"--output_type": Designates the storage type of segmentations if they are stored. [default = 'png']
                              choices=["json", "npy", "npz", "jpg", "png", "dicom-seg"]
    - "-g"/"--gpus": Select specific GPU/CPU to process the input. [default = "0"]
    - "-m"/"--model": Select Model used for inference. [default="UNet_ResNet50_default"]
                      choices=["UNet_ResNet50_default"]     
    - "-s"/"--store_seg": "Wether to also store segmentation masks" [default = False]       
</details>
## Foundation

This work builds on the following papers:

> [**Accurate Fine-Grained Segmentation of Human Anatomy in Radiographs via Volumetric Pseudo-Labeling**]()<br>
>**Purpose:** *The interpretation of chest radiographs (CXR) remains a challenge due to ambiguous overlapping structures such as the lungs, heart, and bones hindering the annotation. To address this, we propose a novel method for extracting fine-grained anatomical structures in CXR using pseudo-labeling of three-dimensional computer tomography (CTs). *
>**Methods:** *We created a large-scale dataset of 10,021 thoracic CTs, encompassing 157 labels, and applied an ensemble of 3D anatomy segmentation models to extract anatomical pseudo-labels. These labels were projected onto a two-dimensional plane, resembling CXR, enabling the training of detailed semantic segmentation models without any manual annotation effort.*
>**Results:** *Our resulting segmentation models demonstrated remarkable performance, with a high average model-annotator agreement between two radiologists with mIoU scores of 0.93 and 0.85 for frontal and lateral anatomies, whereas the inter-annotator agreement remained at 0.95 and 0.83 mIoU. Additionally, our anatomical segmentations allowed for the accurate extraction of relevant explainable medical features such as the Cardio-Thoracic-Ratio.*
>**Conclusion:** *Our method of volumetric pseudo-labeling paired with CT projection offers a promising approach for detailed anatomical segmentation of CXR with a high agreement with human annotators. This technique can have important clinical implications, particularly in the analysis of various thoracic pathologies.*

> [**Detailed Annotations of Chest X-Rays via CT Projection for Report Understanding**](https://bmvc2022.mpi-inf.mpg.de/58/)<br>
> *In clinical radiology reports, doctors capture important information about the patient's health status. They convey their observations from raw medical imaging data about the inner structures of a patient. As such, formulating reports requires medical experts to possess wide-ranging knowledge about anatomical regions with their normal, healthy appearance as well as the ability to recognize abnormalities. This explicit grasp on both the patient's anatomy and their appearance is missing in current medical image-processing systems as annotations are especially difficult to gather. This renders the models to be narrow experts e.g. for identifying specific diseases. In this work, we recover this missing link by adding human anatomy into the mix and enable the association of content in medical reports to their occurrence in associated imagery (medical phrase grounding). To exploit anatomical structures in this scenario, we present a sophisticated automatic pipeline to gather and integrate human bodily structures from computed tomography datasets, which we incorporate in our PAXRay: A Projected dataset for the segmentation of Anatomical structures in X-Ray data. Our evaluation shows that methods that take advantage of anatomical information benefit heavily in visually grounding radiologists' findings, as our anatomical segmentations allow for up to absolute 50% better grounding results on the OpenI dataset than commonly used region proposals.*


## Citation
If you use this work or dataset, please cite:
```latex
@inproceedings{Seibold_2022_BMVC,
author    = {Constantin Marc Seibold and Simon Reiß and M. Saquib Sarfraz and Matthias A. Fink and Victoria Mayer and Jan Sellner and Moon Sung Kim and Klaus H. Maier-Hein and Jens Kleesiek and Rainer Stiefelhagen},
title     = {Detailed Annotations of Chest X-Rays via CT Projection for Report Understanding},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0058.pdf}
}

@inproceedings{Seibold_2023_CXAS,
author    = {Constantin Seibold, Alexander Jaus, Matthias Fink,
Moon Kim, Simon Reiß, Jens Kleesiek*, Rainer Stiefelhagen*},
title     = {Accurate Fine-Grained Segmentation of Human Anatomy in Radiographs via Volumetric Pseudo-Labeling},
year      = {2023},
}

```