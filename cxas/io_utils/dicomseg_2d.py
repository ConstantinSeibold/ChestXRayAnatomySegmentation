import SimpleITK as sitk
import pydicom
import os
import pydicom_seg
from SimpleITK import ImageSeriesReader
from pydicom import Dataset
from pydicom_seg import MultiClassWriter
import numpy as np


def write_dicom_seg(
                    metainfo: str,
                    # actually a directory, not a full path, though you can adjust it to make it
                    # work with paths
                    dcm_file: str,
                    mask: np.array,
                    out_dir: str,
                    id_label_dict: dict,
                ) -> Dataset:
    """
    :param metainfo: path to metainfo.json
    :param dcm_file: directory to the stored series (or singular file)
    :param mask: segmentation mask
    :return: pydicom dataset containing the dicom seg
    """
    template: Dataset = pydicom_seg.template.from_dcmqi_metainfo(metainfo)
    writer: MultiClassWriter = pydicom_seg.MultiClassWriter(
        template=template,
        inplane_cropping=False,
        skip_empty_slices=True,
        skip_missing_segment=False
    )
    reader: ImageSeriesReader = sitk.ImageSeriesReader()
    dcm_files: tuple = (dcm_file,)
    reader.SetFileNames(dcm_files)
    image = reader.Execute()
    image_data_shape: tuple = sitk.GetArrayFromImage(image).shape

    assert image_data_shape[1:] == mask.shape[1:], f'Mask dimensions {mask.shape} and raw image dimensions {image_data_shape} ' \
                                           f'do not match'
    
    for i in range(mask.shape[0]):
        if mask[i:i+1].sum()==0:
            continue
        segmentation_data = mask[i:i+1]
        segmentation = sitk.GetImageFromArray(segmentation_data)
        segmentation.CopyInformation(image)

        source_images = []
        for file in dcm_files:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            try:
                _ = ds[0x0020, 0x0032]
            except KeyError:
                ds.ImagePositionPatient = [-212.025, -173.925, -0]
                # raise AttributeError(f'ImagePositionPatient missing in {file}')


            try:
                _ = ds[0x0020, 0x0037]
            except KeyError:
                ds.ImageOrientationPatient = [1, -0, 0, -0, 1, -0]
                # raise AttributeError(f'ImageOrientationPatient missing in {file}')

            source_images.append(ds)

        out = writer.write(segmentation, source_images)
        out_path = os.path.join(out_dir, 
                                id_label_dict[str(i)]\
                               +'.dcm'
                               )
        out.save_as(out_path)
