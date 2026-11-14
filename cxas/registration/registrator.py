"""
Main Registrator class that orchestrates the registration pipeline.

Pipeline:
1. Color space detection and correction
2. Rotation detection and correction
3. Segmentation
4. Landmark extraction
5. Affine transform computation
6. Apply registration
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, Optional, Tuple
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, asdict

from .orientation import OrientationDetector
from .landmarks import LandmarkExtractor, LandmarkPair
from .affine import (
    compute_affine_transform,
    apply_affine,
    apply_affine_to_mask,
    get_identity_transform,
    decompose_affine
)
from .reference import load_reference, get_default_reference_landmarks


@dataclass
class RegistrationResult:
    """Result of registration for a single image."""
    filename: str
    registered_image: np.ndarray
    registered_mask: Optional[np.ndarray]
    affine_matrix: np.ndarray
    metadata: Dict
    success: bool
    error: Optional[str] = None


class Registrator:
    """
    Main registration class that orchestrates the full pipeline.

    Handles:
    - Orientation/color detection and correction
    - Segmentation
    - Landmark extraction
    - Affine registration
    """

    def __init__(
        self,
        model,
        reference_path: Optional[str] = None,
        do_correction: bool = True
    ):
        """
        Initialize the registrator.

        Args:
            model: CXAS model for segmentation and feature extraction
            reference_path: Path to reference features (.npz). If None, uses default.
            do_correction: Whether to perform orientation/color correction
        """
        self.model = model
        self.do_correction = do_correction

        # Load reference
        self.reference = load_reference(reference_path)
        self.reference_landmarks = get_default_reference_landmarks(self.reference)

        # Initialize components
        if do_correction:
            self.orientation_detector = OrientationDetector(self.reference)
            device = self._get_device()
            self.orientation_detector.to(device)

        self.landmark_extractor = LandmarkExtractor()

    def _get_device(self) -> torch.device:
        """Get the device the model is on."""
        gpus = self.model.gpus
        if isinstance(gpus, str):
            if gpus == "cpu" or gpus == "mps":
                return torch.device(gpus)
            elif gpus.startswith("cuda:"):
                return torch.device(gpus)
            else:
                return torch.device("cpu")
        elif isinstance(gpus, list) and len(gpus) > 0:
            first_gpu = gpus[0]
            if first_gpu == "cpu" or first_gpu == "mps":
                return torch.device(first_gpu)
            elif first_gpu.startswith("cuda:"):
                return torch.device(first_gpu)
            elif isinstance(first_gpu, int):
                return torch.device(f"cuda:{first_gpu}")
            else:
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    def _extract_pooled_features(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """Extract pooled features from backbone."""
        with torch.no_grad():
            backbone = getattr(self.model.model, "backbone", None)
            if backbone is None:
                # The orientation/colour reference (reference_features.npz) holds UNet features, so the check runs on
                # the UNet backbone whichever model segments; loaded once, only when the segmentor is not the UNet.
                backbone = self._unet_backbone(data.device)
            backbone_dict = backbone(data)
            return backbone_dict["feats_pooled"]

    def _unet_backbone(self, device):
        if getattr(self, "_unet_bb", None) is None:
            from ..models import get_model
            gpus = "cpu" if device.type == "cpu" else str(device.index or 0)
            self._unet_bb = get_model("UNet_ResNet50_default", gpus).backbone.eval()
        return self._unet_bb

    def _correct_orientation(
        self,
        data: torch.Tensor,
        orig_image: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        """
        Detect and correct orientation/color issues.

        Returns corrected data, corrected original image, and metadata.
        """
        metadata = {
            "color_detected": "normal",
            "color_corrected": False,
            "rotation_detected": 0,
            "rotation_corrected": False
        }

        if not self.do_correction:
            return data, orig_image, metadata

        # Stage 1: Color detection
        features = self._extract_pooled_features(data)[0]
        color_mode, needs_inversion = self.orientation_detector.detect_color(features)
        metadata["color_detected"] = color_mode

        if needs_inversion:
            data = OrientationDetector.invert_image(data)
            orig_image = OrientationDetector.invert_image_numpy(orig_image)
            metadata["color_corrected"] = True

        # Stage 2: Rotation detection (on color-corrected image)
        features = self._extract_pooled_features(data)[0]
        detected_angle, correction_angle = self.orientation_detector.detect_rotation(features)
        metadata["rotation_detected"] = detected_angle

        if correction_angle != 0:
            data = OrientationDetector.rotate_image(data, correction_angle)
            orig_image = OrientationDetector.rotate_image_numpy(orig_image, correction_angle)
            metadata["rotation_corrected"] = True
            metadata["rotation_correction_angle"] = correction_angle

        return data, orig_image, metadata

    def register_single(
        self,
        file_dict: Dict,
        reference_landmarks: Optional[LandmarkPair] = None,
        save_mask: bool = False
    ) -> RegistrationResult:
        """
        Register a single image.

        Args:
            file_dict: Dictionary from FileLoader with 'data', 'orig_data', 'filename', 'file_size'
            reference_landmarks: Target landmarks. If None, uses reference from file.
            save_mask: Whether to include registered mask in result

        Returns:
            RegistrationResult
        """
        # Handle batch format from dataloader (lists) vs single item format
        filename = file_dict["filename"]
        if isinstance(filename, list):
            filename = filename[0]

        # the folder loader keeps images on the CPU; run on the model's device
        data = file_dict["data"].to(self._get_device())
        orig_data = file_dict["orig_data"]
        if isinstance(orig_data, list):
            orig_data = orig_data[0]

        file_size = file_dict["file_size"]
        if isinstance(file_size, list):
            file_size = file_size[0]

        # Transpose orig_data to (H, W, C) if needed
        if orig_data.shape[0] == 3:
            orig_image = np.transpose(orig_data, (1, 2, 0))
        else:
            orig_image = orig_data

        try:
            # Step 1: Orientation correction
            data_corrected, orig_corrected, orientation_meta = self._correct_orientation(
                data, orig_image
            )

            # Step 2: Segmentation on corrected image
            with torch.no_grad():
                predictions = self.model.model({
                    "data": data_corrected,
                    # the query-based models predict from the original image
                    "orig_data": [np.ascontiguousarray(np.transpose(orig_corrected, (2, 0, 1)))],
                    "filename": [filename],
                    "file_size": [file_size]
                })

            # Segmentation on the 512x512 grid of the model input (M2F_anatomy_v2 predicts at native resolution)
            from ..segmentor import to_feature_grid
            segmentation = to_feature_grid(predictions["segmentation_preds"][0], data_corrected[0])

            # Step 3: Landmark extraction
            landmarks = self.landmark_extractor.extract(segmentation)

            # Use provided reference or default
            ref_landmarks = reference_landmarks or self.reference_landmarks

            if not landmarks.valid:
                return RegistrationResult(
                    filename=filename,
                    registered_image=orig_corrected,
                    registered_mask=segmentation if save_mask else None,
                    affine_matrix=get_identity_transform(),
                    metadata={
                        **orientation_meta,
                        "landmarks": landmarks.to_dict(),
                        "registration_success": False,
                        "error": "T4 or T10 landmarks not detected"
                    },
                    success=False,
                    error="T4 or T10 landmarks not detected"
                )

            if ref_landmarks is None:
                return RegistrationResult(
                    filename=filename,
                    registered_image=orig_corrected,
                    registered_mask=segmentation if save_mask else None,
                    affine_matrix=get_identity_transform(),
                    metadata={
                        **orientation_meta,
                        "landmarks": landmarks.to_dict(),
                        "registration_success": False,
                        "error": "No reference landmarks available"
                    },
                    success=False,
                    error="No reference landmarks available"
                )

            # Scale landmarks from 512x512 to original size
            scale_y = orig_corrected.shape[0] / 512.0
            scale_x = orig_corrected.shape[1] / 512.0

            from .landmarks import Landmark
            scaled_landmarks = LandmarkPair(
                t4=Landmark(
                    name="T4",
                    x=landmarks.t4.x * scale_x,
                    y=landmarks.t4.y * scale_y,
                    valid=landmarks.t4.valid
                ),
                t10=Landmark(
                    name="T10",
                    x=landmarks.t10.x * scale_x,
                    y=landmarks.t10.y * scale_y,
                    valid=landmarks.t10.valid
                )
            )

            # Scale reference landmarks similarly
            scaled_ref = LandmarkPair(
                t4=Landmark(
                    name="T4",
                    x=ref_landmarks.t4.x * scale_x,
                    y=ref_landmarks.t4.y * scale_y,
                    valid=ref_landmarks.t4.valid
                ),
                t10=Landmark(
                    name="T10",
                    x=ref_landmarks.t10.x * scale_x,
                    y=ref_landmarks.t10.y * scale_y,
                    valid=ref_landmarks.t10.valid
                )
            )

            # Step 4: Compute affine transform
            affine_matrix, affine_success = compute_affine_transform(
                scaled_landmarks, scaled_ref
            )

            if not affine_success:
                return RegistrationResult(
                    filename=filename,
                    registered_image=orig_corrected,
                    registered_mask=segmentation if save_mask else None,
                    affine_matrix=get_identity_transform(),
                    metadata={
                        **orientation_meta,
                        "landmarks": landmarks.to_dict(),
                        "registration_success": False,
                        "error": "Failed to compute affine transform"
                    },
                    success=False,
                    error="Failed to compute affine transform"
                )

            # Step 5: Apply registration
            registered_image = apply_affine(orig_corrected, affine_matrix)

            registered_mask = None
            if save_mask:
                # Resize segmentation to original size first
                seg_resized = F.interpolate(
                    torch.tensor(segmentation).float().unsqueeze(0),
                    size=(orig_corrected.shape[0], orig_corrected.shape[1]),
                    mode='nearest'
                )[0].numpy().astype(bool)
                registered_mask = apply_affine_to_mask(seg_resized, affine_matrix)

            # Build metadata
            transform_params = decompose_affine(affine_matrix)
            metadata = {
                **orientation_meta,
                "landmarks": landmarks.to_dict(),
                "reference_landmarks": ref_landmarks.to_dict(),
                "transform": transform_params,
                "registration_success": True
            }

            return RegistrationResult(
                filename=filename,
                registered_image=registered_image,
                registered_mask=registered_mask,
                affine_matrix=affine_matrix,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            return RegistrationResult(
                filename=filename,
                registered_image=orig_image,
                registered_mask=None,
                affine_matrix=get_identity_transform(),
                metadata={"error": str(e)},
                success=False,
                error=str(e)
            )


def save_registration_result(
    result: RegistrationResult,
    output_dir: str,
    save_mask: bool = False
) -> None:
    """
    Save registration result to disk.

    Outputs:
    - {name}_registered.png - Registered image
    - {name}_metadata.json - Metadata including transform params
    - {name}_affine.txt - 2x3 affine matrix
    - {name}_registered_mask.npy - (optional) Registered segmentation

    Args:
        result: RegistrationResult to save
        output_dir: Output directory
        save_mask: Whether to save the registered mask
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get base name
    name = Path(result.filename).stem

    # Save registered image
    img_path = os.path.join(output_dir, f"{name}_registered.png")
    if result.registered_image.dtype != np.uint8:
        img_to_save = (result.registered_image * 255).astype(np.uint8) \
            if result.registered_image.max() <= 1 else result.registered_image.astype(np.uint8)
    else:
        img_to_save = result.registered_image
    Image.fromarray(img_to_save).save(img_path)

    # Save metadata
    meta_path = os.path.join(output_dir, f"{name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(result.metadata, f, indent=2)

    # Save affine matrix
    affine_path = os.path.join(output_dir, f"{name}_affine.txt")
    np.savetxt(affine_path, result.affine_matrix)

    # Save mask if requested
    if save_mask and result.registered_mask is not None:
        mask_path = os.path.join(output_dir, f"{name}_registered_mask.npy")
        np.save(mask_path, result.registered_mask)
