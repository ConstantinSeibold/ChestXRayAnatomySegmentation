import sys, os
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "docs/shortREADME.rst").read_text()

setup(
    name="cxas",
    version="0.0.18",
    description="Segmentation of 159 anatomical classes for Chest X-Rays.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation",
    author="Constantin Seibold",
    author_email="constantin.seibold2@uk-essen.de",
    python_requires=">=3.9",
    packages=[
        "cxas",
        "cxas.io_utils",
        "cxas.models",
        "cxas.extraction",
        "cxas.models.UNet",
        "cxas.registration",
    ],
    package_data={"cxas": ["data/*.json", "data/*.npz"]},
    install_requires=[
        "cython",
        "torch",
        "gdown",
        "torchvision",
        "numpy",
        "SimpleITK",
        "pydicom>=2.4.5,<3",
        "pydicom_seg==0.4.1",
        "scikit-image",
        "scikit-learn",
        "opencv-python",
        "colorcet",
        "pycocotools",
        "pandas",
        "tqdm",
    ],
    extras_require={
        "radiomics": ["pyradiomics==3.0.1"],
    },
    zip_safe=False,
    keywords="chest x-ray anatomy segmntation pytorch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
        "License :: Free for non-commercial use",
    ],
    scripts=[
        "bin/cxas",
        "bin/cxas_feat_extract",
        "bin/cxas_segment",
        "bin/cxas_register",
    ],
)
