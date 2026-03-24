"""
uklidar.py
==========
Superpoint Transformer dataset class for UK colourised LiDAR point
clouds (EA National LIDAR Programme, Bluesky, MLS, etc.).

This module provides the reader and dataset class needed to run
SPT inference on UK LiDAR data using DALES or KITTI-360 pretrained
checkpoints.

Place this file in:
    superpoint_transformer/src/datasets/uklidar.py

Then create the corresponding Hydra configs (see uklidar.yaml).

The reader function ``read_uklidar_tile()`` parses LAS/LAZ files via
PDAL and returns a ``torch_geometric.data.Data`` object with:

    pos       — (N, 3) float32: X, Y, Z coordinates
    rgb       — (N, 3) float32: R, G, B normalised to [0, 1]
    intensity — (N, 1) float32: normalised to [0, 1]
    y         — (N,) int64: semantic labels remapped to target space

For inference on unseen data, all labels are set to the void class
(``num_classes``), which SPT excludes from metrics computation.

Usage
-----
Place colourised LAS/LAZ files into::

    data/uklidar/raw/test/*.laz

Then run::

    python src/eval.py experiment=semantic/uklidar \\
        ckpt_path=/path/to/dales_checkpoint.ckpt

Author: James (Ordnance Survey)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data as PyGData

# SPT's extended Data class with .show() visualisation
try:
    from src.data import Data
except ImportError:
    # Fallback if not running inside the SPT repo
    Data = PyGData

try:
    import pdal
    HAS_PDAL = True
except Exception as _pdal_err:
    HAS_PDAL = False
    # Store the actual error for diagnostic reporting
    _PDAL_IMPORT_ERROR = _pdal_err

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

from src.datasets.base import BaseDataset

log = logging.getLogger(__name__)


# ===========================================================================
# DALES-compatible class definitions (default target)
# ===========================================================================

DALES_NUM_CLASSES = 8

# class_names must have num_classes + 1 entries; the last is the void class
DALES_CLASS_NAMES = [
    'Ground',
    'Vegetation',
    'Cars',
    'Trucks',
    'Power lines',
    'Fences',
    'Poles',
    'Buildings',
    'Unknown',          # void / unlabelled (index == num_classes)
]

# class_colors must also have num_classes + 1 entries
DALES_CLASS_COLORS = [
    [128,  64,   0],   # ground — brown
    [  0, 128,   0],   # vegetation — green
    [255,   0,   0],   # cars — red
    [255, 128,   0],   # trucks — orange
    [255, 255,   0],   # power_lines — yellow
    [128, 128, 128],   # fences — grey
    [  0,   0, 255],   # poles — blue
    [255,   0, 255],   # buildings — magenta
    [ 50,  50,  50],   # unknown / void — dark grey
]

DALES_STUFF_CLASSES = [0, 1]  # Ground, Vegetation (same as DALES)

# ASPRS → DALES label mapping
# Applied when reading classified UK LiDAR data.
# For inference on unclassified data, all points get void label (8).
ASPRS_TO_DALES = {
    0: 8,   # Created/Never classified → void
    1: 8,   # Unclassified → void
    2: 0,   # Ground → ground
    3: 1,   # Low Vegetation → vegetation
    4: 1,   # Medium Vegetation → vegetation
    5: 1,   # High Vegetation → vegetation
    6: 7,   # Building → buildings
    7: 8,   # Low Point (Noise) → void
    9: 8,   # Water → void
    17: 8,  # Bridge Deck → void
    18: 8,  # High Noise → void
    20: 8,  # Permanent Structure → void
}

# DALES predictions → ASPRS codes (reverse mapping)
DALES_TO_ASPRS = {
    0: 2, 1: 5, 2: 1, 3: 1, 4: 14, 5: 13, 6: 15, 7: 6, 8: 1,
}

# Backwards-compatible aliases
ASPRS_TO_TARGET = ASPRS_TO_DALES
UKLIDAR_NUM_CLASSES = DALES_NUM_CLASSES
UKLIDAR_CLASS_NAMES = DALES_CLASS_NAMES
UKLIDAR_CLASS_COLORS = DALES_CLASS_COLORS
UKLIDAR_STUFF_CLASSES = DALES_STUFF_CLASSES


# ===========================================================================
# KITTI-360-compatible class definitions
# ===========================================================================

KITTI360_NUM_CLASSES = 19

KITTI360_CLASS_NAMES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

KITTI360_CLASS_COLORS = [
    [128, 64, 128],   [244, 35, 232],  [70, 70, 70],
    [102, 102, 156],  [190, 153, 153], [153, 153, 153],
    [250, 170, 30],   [220, 220, 0],   [107, 142, 35],
    [152, 251, 152],  [70, 130, 180],  [220, 20, 60],
    [255, 0, 0],      [0, 0, 142],     [0, 0, 70],
    [0, 60, 100],     [0, 80, 100],    [0, 0, 230],
    [119, 11, 32],
]

ASPRS_TO_KITTI360 = {
    0: 19,   # Never classified -> void
    1: 19,   # Unassigned -> void
    2: 9,    # Ground -> terrain
    3: 8,    # Low Vegetation -> vegetation
    4: 8,    # Medium Vegetation -> vegetation
    5: 8,    # High Vegetation -> vegetation
    6: 2,    # Building -> building
    7: 19,   # Low Point -> void
    9: 19,   # Water -> void
    10: 19,  # Rail -> void (no trainId in KITTI-360)
    11: 0,   # Road Surface -> road
    13: 19,  # Wire - Guard -> void
    14: 19,  # Wire - Conductor -> void
    15: 5,   # Transmission Tower -> pole
    16: 19,  # Wire-Structure Connector -> void
    17: 19,  # Bridge Deck -> void
    18: 19,  # High Noise -> void
}

KITTI360_TO_ASPRS = {
    0: 11,   # road -> Road Surface
    1: 11,   # sidewalk -> Road Surface
    2: 6,    # building -> Building
    3: 64,   # wall -> User Definable
    4: 65,   # fence -> User Definable
    5: 66,   # pole -> User Definable
    6: 67,   # traffic light -> User Definable
    7: 68,   # traffic sign -> User Definable
    8: 5,    # vegetation -> High Vegetation
    9: 2,    # terrain -> Ground
    10: 69,  # sky -> User Definable
    11: 70,  # person -> User Definable
    12: 71,  # rider -> User Definable
    13: 72,  # car -> User Definable
    14: 73,  # truck -> User Definable
    15: 74,  # bus -> User Definable
    16: 75,  # train -> User Definable
    17: 76,  # motorcycle -> User Definable
    18: 77,  # bicycle -> User Definable
}


# ===========================================================================
# Target configuration registry
# ===========================================================================

TARGET_CONFIGS = {
    "dales": {
        "num_classes": DALES_NUM_CLASSES,
        "class_names": DALES_CLASS_NAMES,
        "class_colors": DALES_CLASS_COLORS,
        "asprs_to_target": ASPRS_TO_DALES,
        "target_to_asprs": DALES_TO_ASPRS,
        "void_label": 8,
        "spt_model": "spt-64",
        "tiling": "xy",
    },
    "kitti360": {
        "num_classes": KITTI360_NUM_CLASSES,
        "class_names": KITTI360_CLASS_NAMES,
        "class_colors": KITTI360_CLASS_COLORS,
        "asprs_to_target": ASPRS_TO_KITTI360,
        "target_to_asprs": KITTI360_TO_ASPRS,
        "void_label": 19,
        "spt_model": "spt-128",
        "tiling": "pc",
    },
}


def get_target_config(target="dales"):
    """Return the full config dict for a target dataset."""
    if target not in TARGET_CONFIGS:
        raise ValueError(
            "Unknown target '{}'. Choose from: {}".format(
                target, list(TARGET_CONFIGS.keys()))
        )
    return TARGET_CONFIGS[target]


# ===========================================================================
# Reader Function
# ===========================================================================

def read_uklidar_tile(
    filepath: str | Path,
    target: str = "dales",
    remap: dict[int, int] | None = None,
    void_label: int | None = None,
    label_all_void: bool = False,
) -> Data:
    """Read a colourised LAS/LAZ file and return an SPT-compatible
    Data object.

    Tries PDAL first, falls back to laspy if PDAL import fails.
    Both are available in the spt conda environment.

    Args:
        filepath:       Path to the LAS/LAZ file.
        target:         Target dataset config ("dales" or "kitti360").
        remap:          ASPRS → target label dict.  None = use target default.
        void_label:     Label code for void/unlabeled points.  None = use
                        target default.
        label_all_void: If True, set all labels to void (for inference
                        on data without ground truth).

    Returns:
        Data object with pos, rgb, intensity, y attributes.
    """
    filepath = Path(filepath)
    cfg = get_target_config(target)
    if remap is None:
        remap = cfg["asprs_to_target"]
    if void_label is None:
        void_label = cfg["void_label"]

    # --- Read point cloud (PDAL or laspy) ---
    if HAS_PDAL:
        points_dict = _read_with_pdal(filepath)
    elif HAS_LASPY:
        log.info("PDAL not available (reason: %s), using laspy fallback",
                 _PDAL_IMPORT_ERROR if not HAS_PDAL else "N/A")
        points_dict = _read_with_laspy(filepath)
    else:
        msg = "Neither PDAL nor laspy is available."
        if not HAS_PDAL:
            msg += "\n  PDAL import error: {}".format(_PDAL_IMPORT_ERROR)
        msg += "\n  Install one via: conda install -c conda-forge python-pdal"
        msg += "\n  Or: pip install laspy[laszip]"
        raise ImportError(msg)

    n = points_dict["n"]
    pos = points_dict["pos"]
    rgb = points_dict["rgb"]
    intensity = points_dict["intensity"]
    raw_labels = points_dict["classification"]

    # --- Labels ---
    if label_all_void or raw_labels is None:
        y = np.full(n, void_label, dtype=np.int64)
    else:
        max_code = max(max(remap.keys()), int(raw_labels.max())) + 1
        lut = np.full(max_code, void_label, dtype=np.int64)
        for src, dst in remap.items():
            if src < max_code:
                lut[src] = dst
        y = lut[np.clip(raw_labels.astype(np.int64), 0, max_code - 1)]

    # --- Build Data ---
    data = Data(
        pos=torch.from_numpy(pos),
        rgb=torch.from_numpy(rgb),
        intensity=torch.from_numpy(intensity),
        y=torch.from_numpy(y),
    )

    return data


def _normalise_rgb(r, g, b):
    """Normalise RGB arrays to [0, 1] float32."""
    max_val = max(r.max(), g.max(), b.max())
    if max_val > 255:
        divisor = 65535.0
    elif max_val > 1:
        divisor = 255.0
    else:
        divisor = 1.0
    return np.column_stack([r, g, b]).astype(np.float32) / divisor


def _normalise_intensity(raw_i):
    """Normalise intensity to [0, 1] float32, shape (N, 1)."""
    raw_i = raw_i.astype(np.float32)
    i_max = raw_i.max()
    if i_max > 0:
        return (raw_i / i_max).reshape(-1, 1)
    return np.zeros((len(raw_i), 1), dtype=np.float32)


def _read_with_pdal(filepath):
    """Read LAS/LAZ via PDAL, return a dict of numpy arrays."""
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [{"type": "readers.las", "filename": str(filepath)}]
    }))
    pipeline.execute()
    points = pipeline.arrays[0]
    dim_names = list(points.dtype.names)
    n = len(points)

    pos = np.column_stack([
        points["X"].astype(np.float32),
        points["Y"].astype(np.float32),
        points["Z"].astype(np.float32),
    ])

    if "Red" in dim_names and "Green" in dim_names and "Blue" in dim_names:
        rgb = _normalise_rgb(points["Red"], points["Green"], points["Blue"])
    else:
        rgb = np.zeros((n, 3), dtype=np.float32)

    if "Intensity" in dim_names:
        intensity = _normalise_intensity(points["Intensity"])
    else:
        intensity = np.zeros((n, 1), dtype=np.float32)

    classification = None
    if "Classification" in dim_names:
        classification = points["Classification"]

    return {"n": n, "pos": pos, "rgb": rgb,
            "intensity": intensity, "classification": classification}


def _read_with_laspy(filepath):
    """Read LAS/LAZ via laspy, return a dict of numpy arrays."""
    las = laspy.read(str(filepath))
    n = len(las.points)

    pos = np.column_stack([
        np.asarray(las.x, dtype=np.float32),
        np.asarray(las.y, dtype=np.float32),
        np.asarray(las.z, dtype=np.float32),
    ])

    # RGB — laspy exposes these as point_format dimensions
    try:
        rgb = _normalise_rgb(
            np.asarray(las.red),
            np.asarray(las.green),
            np.asarray(las.blue),
        )
    except Exception:
        rgb = np.zeros((n, 3), dtype=np.float32)

    # Intensity
    try:
        intensity = _normalise_intensity(np.asarray(las.intensity))
    except Exception:
        intensity = np.zeros((n, 1), dtype=np.float32)

    # Classification
    classification = None
    try:
        classification = np.asarray(las.classification)
    except Exception:
        pass

    return {"n": n, "pos": pos, "rgb": rgb,
            "intensity": intensity, "classification": classification}


# ===========================================================================
# Dataset Class
# ===========================================================================

__all__ = ['UKLidarDataset']


class UKLidarDataset(BaseDataset):
    """Dataset class for UK colourised LiDAR point clouds.

    Inherits from ``src.datasets.base.BaseDataset`` to integrate with
    SPT's preprocessing, transform, and evaluation pipeline.

    Directory structure::

        data/uklidar/
        ├── raw/
        │   ├── train/           # Training tiles (optional)
        │   │   └── tile_A.laz
        │   ├── val/             # Validation tiles (optional)
        │   │   └── tile_B.laz
        │   └── test/            # Test/inference tiles
        │       └── tile_C.laz
        └── processed/
            └── <hash>/          # Auto-generated by SPT preprocessing
                └── tile_C.h5

    For inference-only usage, place all files in ``raw/test/``.
    """

    # Whether labels are available for inference data
    label_all_void: bool = True

    @property
    def class_names(self) -> List[str]:
        """List of string names for dataset classes. Must be one-item
        larger than ``num_classes``, with the last entry being the
        void / unlabelled class.
        """
        return UKLIDAR_CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return UKLIDAR_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        """'Stuff' classes (Ground, Vegetation) — same convention as
        DALES.
        """
        return UKLIDAR_STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        return UKLIDAR_CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> Dict[str, List[str]]:
        """Discover cloud IDs from the raw directory structure.

        Returns dict with train/val/test splits populated from the
        corresponding subdirectories.  Cloud IDs are just the file
        stems (no split prefix) — the split prefix is handled by
        ``id_to_relative_raw_path``.
        """
        raw_dir = Path(self.raw_dir)
        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        for split in splits:
            split_dir = raw_dir / split
            if split_dir.is_dir():
                laz = sorted(split_dir.glob("*.laz"))
                las = sorted(split_dir.glob("*.las"))
                ply = sorted(split_dir.glob("*.ply"))
                all_files = sorted(
                    set(laz + las + ply), key=lambda p: p.name,
                )
                splits[split] = [f.stem for f in all_files]

        log.info(
            "UKLidar splits: train=%d, val=%d, test=%d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

        return splits

    def download_dataset(self) -> None:
        """UKLidar data is user-provided — no automatic download."""
        log.info(
            "UKLidar does not support automatic download.\n"
            "Place your LAS/LAZ files in:\n"
            "  %s/{train,val,test}/*.laz\n",
            self.raw_dir,
        )

    def id_to_relative_raw_path(self, id: str) -> str:
        """Map a cloud id to its relative raw path under ``raw_dir``.

        We need to find the actual file because extensions may be
        .laz, .las, or .ply.
        """
        base_id = self.id_to_base_id(id)

        # Determine which split this id belongs to
        for split, ids in self.all_cloud_ids.items():
            if id in ids:
                # Resolve 'trainval' and 'val' to the correct raw dir
                raw_split = split
                if raw_split == 'trainval':
                    raw_split = 'train'
                split_dir = Path(self.raw_dir) / raw_split
                for ext in ('.laz', '.las', '.ply'):
                    candidate = split_dir / (base_id + ext)
                    if candidate.is_file():
                        return str(Path(raw_split) / (base_id + ext))
                # Fallback: return .laz path even if not found yet
                return str(Path(raw_split) / (base_id + '.laz'))

        raise ValueError(f"Unknown cloud id '{id}'")

    def processed_to_raw_path(self, processed_path: str) -> str:
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        import os
        stage, hash_dir, cloud_id = \
            os.path.splitext(processed_path)[0].split(os.sep)[-3:]

        raw_split = 'train' if stage in ['trainval', 'val'] else stage
        base_cloud_id = self.id_to_base_id(cloud_id)

        split_dir = Path(self.raw_dir) / raw_split
        for ext in ('.laz', '.las', '.ply'):
            candidate = split_dir / (base_cloud_id + ext)
            if candidate.is_file():
                return str(candidate)

        return str(split_dir / (base_cloud_id + '.laz'))

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        """Read a single raw point cloud file.

        This is the method that SPT's preprocessing pipeline calls.
        It dispatches to ``read_uklidar_tile()`` for LAS/LAZ files.
        """
        raw_cloud_path = Path(raw_cloud_path)

        if raw_cloud_path.suffix.lower() in (".las", ".laz"):
            data = read_uklidar_tile(
                raw_cloud_path,
                label_all_void=self.label_all_void,
            )
        elif raw_cloud_path.suffix.lower() == ".ply":
            data = self._read_ply(raw_cloud_path)
        else:
            raise ValueError(
                f"Unsupported file format: {raw_cloud_path.suffix}"
            )

        return data

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        └── raw/
            └── {{train, val, test}}/
                └── {{tile_name}}.laz
            """

    @staticmethod
    def _read_ply(filepath):
        """Read an SPT-format PLY file."""
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError(
                "plyfile required for PLY reading: pip install plyfile"
            )

        plydata = PlyData.read(str(filepath))
        vertex = plydata["vertex"]

        pos = np.column_stack([
            vertex["x"], vertex["y"], vertex["z"],
        ]).astype(np.float32)

        # Try standard names
        if "red" in vertex.data.dtype.names:
            rgb = np.column_stack([
                vertex["red"], vertex["green"], vertex["blue"],
            ]).astype(np.float32)
            # Normalise if needed
            if rgb.max() > 1.0:
                rgb = rgb / (65535.0 if rgb.max() > 255 else 255.0)
        else:
            rgb = np.zeros((len(pos), 3), dtype=np.float32)

        if "intensity" in vertex.data.dtype.names:
            raw_i = vertex["intensity"].astype(np.float32)
            i_max = raw_i.max()
            intensity = (raw_i / i_max if i_max > 0
                         else np.zeros_like(raw_i)).reshape(-1, 1)
        else:
            intensity = np.zeros((len(pos), 1), dtype=np.float32)

        if "scalar_Classification" in vertex.data.dtype.names:
            y = vertex["scalar_Classification"].astype(np.int64)
        elif "label" in vertex.data.dtype.names:
            y = vertex["label"].astype(np.int64)
        else:
            y = np.full(len(pos), UKLIDAR_NUM_CLASSES, dtype=np.int64)

        return Data(
            pos=torch.from_numpy(pos),
            rgb=torch.from_numpy(rgb),
            intensity=torch.from_numpy(intensity),
            y=torch.from_numpy(y),
        )
