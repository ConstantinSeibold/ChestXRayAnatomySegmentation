import SimpleITK as sitk
import pydicom
import os
from SimpleITK import ImageSeriesReader
from pydicom import Dataset
from pydicom_seg import MultiClassWriter
import numpy as np
from pydicom_seg.template import from_dcmqi_metainfo


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
    Writes DICOM Segmentation objects.

    :param metainfo: Path to metainfo.json.
    :param dcm_file: Directory to the stored series (or singular file).
    :param mask: Segmentation mask.
    :param out_dir: Output directory for DICOM Segmentation objects.
    :param id_label_dict: Dictionary mapping label indices to identifiers.
    :return: Pydicom dataset containing the DICOM SEG.
    """
    # Generate DICOM SEG template from metadata information
    template: Dataset = from_dcmqi_metainfo(metainfo)
    
    # Initialize MultiClassWriter for DICOM SEG creation
    writer: MultiClassWriter = MultiClassWriter(
        template=template,
        inplane_cropping=False,
        skip_empty_slices=True,
        skip_missing_segment=False
    )
    
    # Read DICOM series
    reader: ImageSeriesReader = sitk.ImageSeriesReader()
    dcm_files: tuple = (dcm_file,)
    reader.SetFileNames(dcm_files)
    image = reader.Execute()
    
    # Get shape of image data from SimpleITK image
    image_data_shape: tuple = sitk.GetArrayFromImage(image).shape
    
    # Ensure mask dimensions match image dimensions
    assert image_data_shape[1:] == mask.shape[1:], \
        f'Mask dimensions {mask.shape} and raw image dimensions {image_data_shape} do not match'
    
    # Iterate over slices in the mask
    for i in range(mask.shape[0]):
        # Skip if the mask slice is empty
        if mask[i:i+1].sum() == 0:
            continue
        
        # Extract segmentation data for the current slice
        segmentation_data = mask[i:i+1]
        segmentation = sitk.GetImageFromArray(segmentation_data)
        segmentation.CopyInformation(image)
        
        # Extract DICOM metadata from source images
        source_images = []
        for file in dcm_files:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            # Set default values if ImagePositionPatient or ImageOrientationPatient are missing
            try:
                _ = ds[0x0020, 0x0032]
            except KeyError:
                ds.ImagePositionPatient = [-212.025, -173.925, -0]
            
            try:
                _ = ds[0x0020, 0x0037]
            except KeyError:
                ds.ImageOrientationPatient = [1, -0, 0, -0, 1, -0]
            
            source_images.append(ds)
        
        # Write DICOM SEG and save to output directory
        out = writer.write(segmentation, source_images)
        out_path = os.path.join(out_dir, id_label_dict[str(i)] + '.dcm')
        out.save_as(out_path)
