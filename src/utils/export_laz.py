"""Utilities for exporting NAG data and inference results to LAZ files.

Usage examples
--------------
**From a saved HDF5 prediction file** (produced by ``predict_step``)::

    from src.data import NAG
    from src.utils.export_laz import nag_to_laz

    nag = NAG.load("predictions/predict/0/batch_0.h5")
    nag_to_laz(nag, "output.laz")

**From a NAG object after inference** (e.g. inside a callback)::

    nag_to_laz(nag, "output.laz")

**Exporting only level-0 (voxel) data with predictions**::

    nag_to_laz(nag, "output.laz", level=0)

**Command-line**::

    python -m src.utils.export_laz predictions/predict/0/batch_0.h5 output.laz
"""

import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ['nag_to_laz', 'data_to_laz']


def _require_laspy():
    try:
        import laspy
        return laspy
    except ImportError:
        raise ImportError(
            "laspy is required for LAZ export. "
            "Install it with: pip install laspy[laszip]"
        )


def data_to_laz(data, path, extra_attrs=None):
    """Export a single ``Data`` object (one NAG level) to a LAZ file.

    Parameters
    ----------
    data : src.data.Data
        Point-level data with at least a ``pos`` attribute of shape (N, 3).
    path : str or Path
        Output file path. Use ``.laz`` for compressed or ``.las`` for
        uncompressed.
    extra_attrs : list[str], optional
        Additional ``data`` attributes to write as extra scalar dimensions.
        Each must be a 1-D tensor of length N.
    """
    laspy = _require_laspy()
    import torch

    path = Path(path)
    pos = data.pos
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    pos = pos.astype(np.float64)

    n = pos.shape[0]

    # Use point format 2 (XYZ + RGB) if RGB is available, else format 0
    has_rgb = getattr(data, 'rgb', None) is not None
    point_format = 2 if has_rgb else 0

    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.offsets = pos.min(axis=0)
    # Scale chosen for mm precision
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = pos[:, 0]
    las.y = pos[:, 1]
    las.z = pos[:, 2]

    # RGB – stored as [0, 1] floats in SPT, LAS uses uint16
    if has_rgb:
        rgb = data.rgb
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        las.red = (rgb[:, 0] * 65535).astype(np.uint16)
        las.green = (rgb[:, 1] * 65535).astype(np.uint16)
        las.blue = (rgb[:, 2] * 65535).astype(np.uint16)

    # Intensity – stored as [0, 1] floats in SPT, LAS uses uint16
    if getattr(data, 'intensity', None) is not None:
        intensity = data.intensity
        if isinstance(intensity, torch.Tensor):
            intensity = intensity.cpu().numpy()
        intensity = np.squeeze(intensity)
        las.intensity = (np.clip(intensity, 0.0, 1.0) * 65535).astype(
            np.uint16)

    # Semantic predictions → classification field
    if getattr(data, 'semantic_pred', None) is not None:
        pred = data.semantic_pred
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        las.classification = pred.astype(np.uint8)

    # Ground-truth labels → extra scalar dimension
    if getattr(data, 'y', None) is not None:
        y = data.y
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        # y can be a 2D histogram; store argmax if so
        if y.ndim == 2:
            y = y.argmax(axis=1)
        las.add_extra_dim(
            laspy.ExtraBytesParams(name="ground_truth", type=np.uint8))
        las.ground_truth = y.astype(np.uint8)

    # Write any user-requested extra attributes
    for attr_name in (extra_attrs or []):
        val = getattr(data, attr_name, None)
        if val is None:
            log.warning("Attribute %r not found on data, skipping", attr_name)
            continue
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        val = np.squeeze(val)
        if val.shape[0] != n:
            log.warning(
                "Attribute %r has length %d (expected %d), skipping",
                attr_name, val.shape[0], n)
            continue
        # Pick a suitable numpy dtype
        if np.issubdtype(val.dtype, np.floating):
            dtype = np.float64
        elif np.issubdtype(val.dtype, np.integer):
            dtype = np.int32
        else:
            dtype = val.dtype
        las.add_extra_dim(
            laspy.ExtraBytesParams(name=attr_name, type=dtype))
        setattr(las, attr_name, val.astype(dtype))

    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))
    log.info("Exported %d points to %s", n, path)


def nag_to_laz(nag, path, level=0, extra_attrs=None):
    """Export a NAG level to a LAZ file.

    Parameters
    ----------
    nag : src.data.NAG
        The nested acyclic graph (e.g. loaded from an HDF5 prediction file).
    path : str or Path
        Output file path (``.laz`` or ``.las``).
    level : int, optional
        Which hierarchy level to export.  ``0`` exports the voxel-level
        points (finest resolution stored in the NAG).  ``1`` exports the
        superpoint-level data.  Default is ``0``.
    extra_attrs : list[str], optional
        Additional attributes to include as extra LAZ dimensions.
    """
    data = nag[level]
    data_to_laz(data, path, extra_attrs=extra_attrs)


# ---------------------------------------------------------------------------
# CLI entry point: python -m src.utils.export_laz <input.h5> <output.laz>
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Export a saved NAG (HDF5) to a LAZ point cloud file.")
    parser.add_argument(
        "input", type=str,
        help="Path to the input HDF5 file (e.g. predictions/batch_0.h5)")
    parser.add_argument(
        "output", type=str,
        help="Path to the output LAZ file")
    parser.add_argument(
        "--level", type=int, default=0,
        help="NAG hierarchy level to export (default: 0 = voxel level)")
    parser.add_argument(
        "--extra-attrs", nargs="*", default=None,
        help="Additional data attributes to include as extra dimensions")
    args = parser.parse_args()

    from src.data import NAG

    nag = NAG.load(args.input)
    nag_to_laz(nag, args.output, level=args.level, extra_attrs=args.extra_attrs)
    print(f"Done. Wrote {args.output}")
