import sys, os
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "docs/shortREADME.rst").read_text()

setup(
    name="cxas",
    version="0.1.0",
    description="Anatomy segmentation (159+ structures) and named landmarks for chest X-rays, with DICOM SEG/SR output.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation",
    author="Constantin Seibold",
    license="MIT",
    author_email="constantin.seibold2@uk-essen.de",
    python_requires=">=3.10",
    packages=[
        "cxas",
        "cxas.io_utils",
        "cxas.models",
        "cxas.extraction",
        "cxas.models.UNet",
        "cxas.registration",
        "cxas.landmarks",
        "cxas.models.m2f",
    ],
    package_data={"cxas": ["data/*.json", "data/*.npz", "data/adapters/*.json", "landmarks/data/*.json", "landmarks/data/*.csv"]},
    install_requires=[
        "cython",
        "torch",
        "gdown",
        "torchvision",
        "numpy",
        "SimpleITK",
        "pydicom>=3.0.1",
        "highdicom>=0.25",
        "pylibjpeg>=2.0",
        "pylibjpeg-openjpeg>=2.3",
        "safetensors",
        "huggingface_hub",
        # >= 5.9: the models were trained with 5.9; transformers 4.x computes the DINOv3 encoder slightly
        # differently (frontal masks drift, worst structure IoU 0.74)
        "transformers>=5.9",
        "scipy",
        "pyyaml",
        "scikit-image",
        "scikit-learn",
        "opencv-python",
        "colorcet",
        "matplotlib",
        "Pillow",
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
        "License :: OSI Approved :: MIT License",
    ],
    scripts=[
        "bin/cxas",
        "bin/cxas_feat_extract",
        "bin/cxas_segment",
        "bin/cxas_register",
        "bin/cxas_idc",
    ],
)
