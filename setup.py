import sys,os
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "docs/shortREADME.rst").read_text()

setup(name='cxas',
        version='0.0.7',
        description='Segmentation of 159 anatomical classes for Chest X-Rays.',
        long_description = long_description,
        long_description_content_type="text/x-rst",
        url='https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation',
        author='Constantin Seibold',
        author_email='constantin.seibold2@uk-essen.de',
        python_requires='>=3.9',
        packages=['cxas', 'cxas.io_utils', 'cxas.models', 'cxas.extraction', 'cxas.models.UNet'],
        package_data={'cxas':['data/*.json']},
        install_requires=[
            'cython',
            'torch',
            'gdown',
            'torchvision',
            'numpy',
            'SimpleITK',
            'pydicom',
            'pydicom_seg',
            'scikit-image',
            'scikit-learn',
            'opencv-python',
            'colorcet',
            'pycocotools',
            'pandas',
            'tqdm',
        ],
        zip_safe=False,
        keywords="chest x-ray anatomy segmntation pytorch",
        classifiers=[
            'Development Status :: 3 - Alpha', 
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',    
             "License :: Free for non-commercial use",
        ],
        scripts=[
            'bin/cxas', 'bin/cxas_feat_extract', 'bin/cxas_segment',
        ]
    )