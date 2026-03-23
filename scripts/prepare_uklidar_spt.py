#!/usr/bin/env python3
"""
prepare_spt.py
==============
Prepare colourised LAS/LAZ point clouds for Superpoint Transformer (SPT)
inference, targeting DALES and KITTI-360 pretrained checkpoints.

Superpoint Transformer Data Contract
-------------------------------------
SPT's ``read_single_raw_cloud()`` must return a ``torch_geometric.data.Data``
object with:

    ``pos``       — (N, 3) float32 tensor: X, Y, Z coordinates
    ``rgb``       — (N, 3) float32 tensor: R, G, B normalised to [0, 1]
    ``intensity`` — (N, 1) float32 tensor: normalised to [0, 1]
    ``y``         — (N,) int64 tensor: semantic labels in [0, C]
                     where C = num_classes = void/unlabeled class

SPT expects raw data as PLY files in ``data/<dataset>/raw/<split>/``.
This script converts LAS/LAZ files to that format.

Target Datasets
---------------
**DALES** — Aerial LiDAR, 8 semantic classes + void::

    0: ground
    1: vegetation
    2: cars
    3: trucks
    4: power lines
    5: fences
    6: poles
    7: buildings
    8: unlabeled (void)

**KITTI-360** — Mobile mapping, 19 semantic classes::

    0: road,  1: sidewalk,  2: building,  3: wall,  4: fence,
    5: pole,  6: traffic light,  7: traffic sign,  8: vegetation,
    9: terrain,  10: sky,  11: person,  12: rider,  13: car,
    14: truck,  15: bus,  16: train,  17: motorcycle,  18: bicycle
    19: unlabeled (void)

ASPRS → Target Remapping
~~~~~~~~~~~~~~~~~~~~~~~~~
Your colourised UK LiDAR uses ASPRS codes.  This script maps them to
the target dataset's label space::

    ASPRS → DALES (default):
        1 (Unclassified) → 8 (void)
        2 (Ground)       → 0 (ground)
        3 (Low Veg)      → 1 (vegetation)
        5 (High Veg)     → 1 (vegetation)
        6 (Building)     → 7 (buildings)
        9 (Water)        → 8 (void)       [no water in DALES]
        17 (Bridge)      → 8 (void)
        20 (Perm Struct)  → 8 (void)

    ASPRS → KITTI-360:
        1 (Unclassified) → 19 (void)
        2 (Ground)       → 9  (terrain)
        3 (Low Veg)      → 8  (vegetation)
        5 (High Veg)     → 8  (vegetation)
        6 (Building)     → 2  (building)
        9 (Water)        → 19 (void)
        17 (Bridge)      → 19 (void)
        20 (Perm Struct)  → 3  (wall)

Usage
-----
    # Convert for DALES checkpoint (default)
    python prepare_spt.py \\
        --input-dir ./colourised/ \\
        --output-dir ./spt_data/raw/test/ \\
        --target dales

    # Convert for KITTI-360 checkpoint
    python prepare_spt.py \\
        --input-dir ./colourised/ \\
        --output-dir ./spt_data/raw/test/ \\
        --target kitti360

    # Diagnose only — inspect attributes
    python prepare_spt.py \\
        --input-dir ./colourised/ \\
        --diagnose-only

    # Use as a Python module (reader function for SPT)
    from prepare_spt import read_las_cloud
    data = read_las_cloud("tile.laz", target="dales")
    # data.pos, data.rgb, data.intensity, data.y ready for SPT

Dependencies
------------
    pip install pdal python-pdal numpy torch torch_geometric plyfile

Author: James (Ordnance Survey)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pdal
except ImportError:
    sys.exit("ERROR: python-pdal required. pip install pdal")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_spt")


# ===========================================================================
# Target Dataset Definitions
# ===========================================================================

DALES_CLASSES = {
    0: "ground",
    1: "vegetation",
    2: "cars",
    3: "trucks",
    4: "power_lines",
    5: "fences",
    6: "poles",
    7: "buildings",
    8: "unlabeled",  # void
}
DALES_NUM_CLASSES = 8  # void = 8

KITTI360_CLASSES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
    19: "unlabeled",  # void
}
KITTI360_NUM_CLASSES = 19  # void = 19

# ASPRS → DALES label mapping
ASPRS_TO_DALES = {
    0: 8,   # Created/Never classified → void
    1: 8,   # Unclassified → void
    2: 0,   # Ground → ground
    3: 1,   # Low Vegetation → vegetation
    4: 1,   # Medium Vegetation → vegetation
    5: 1,   # High Vegetation → vegetation
    6: 7,   # Building → buildings
    7: 8,   # Low Point (Noise) → void
    9: 8,   # Water → void (no water class in DALES)
    17: 8,  # Bridge Deck → void
    18: 8,  # High Noise → void
    20: 8,  # Reserved / Permanent Structure → void
}

# ASPRS → KITTI-360 label mapping
ASPRS_TO_KITTI360 = {
    0: 19,  # Created/Never classified → void
    1: 19,  # Unclassified → void
    2: 9,   # Ground → terrain
    3: 8,   # Low Vegetation → vegetation
    4: 8,   # Medium Vegetation → vegetation
    5: 8,   # High Vegetation → vegetation
    6: 2,   # Building → building
    7: 19,  # Low Point (Noise) → void
    9: 19,  # Water → void
    17: 19, # Bridge Deck → void
    18: 19, # High Noise → void
    20: 3,  # Permanent Structure → wall
}

TARGET_CONFIGS = {
    "dales": {
        "classes": DALES_CLASSES,
        "num_classes": DALES_NUM_CLASSES,
        "remap": ASPRS_TO_DALES,
        "void_label": 8,
    },
    "kitti360": {
        "classes": KITTI360_CLASSES,
        "num_classes": KITTI360_NUM_CLASSES,
        "remap": ASPRS_TO_KITTI360,
        "void_label": 19,
    },
}


# ===========================================================================
# Label Remapping
# ===========================================================================

def remap_labels(
    labels: np.ndarray,
    remap: dict[int, int],
    void_label: int,
) -> np.ndarray:
    """Remap ASPRS classification codes to target dataset labels.

    Uses a numpy LUT for vectorised remapping.  Unmapped codes default
    to the void label.
    """
    max_code = max(max(remap.keys()), int(labels.max())) + 1
    lut = np.full(max_code, void_label, dtype=np.int64)
    for src, dst in remap.items():
        if src < max_code:
            lut[src] = dst
    return lut[np.clip(labels.astype(np.int64), 0, max_code - 1)]


# ===========================================================================
# Core Reader Function — read_las_cloud()
# ===========================================================================

def read_las_cloud(
    filepath: str | Path,
    target: str = "dales",
    remap_override: Optional[dict[int, int]] = None,
) -> "torch_geometric.data.Data":
    """Read a LAS/LAZ file and return an SPT-compatible Data object.

    This is the **reader function** that can be used directly in an SPT
    dataset class's ``read_single_raw_cloud()`` method.

    The returned ``Data`` object has:

        ``pos``       — (N, 3) float32: X, Y, Z
        ``rgb``       — (N, 3) float32: R, G, B in [0, 1]
        ``intensity`` — (N, 1) float32: in [0, 1]
        ``y``         — (N,) int64: remapped labels

    Args:
        filepath: Path to the LAS/LAZ file.
        target:   Target dataset ("dales" or "kitti360").
        remap_override: Custom ASPRS → target label mapping.

    Returns:
        torch_geometric.data.Data object.
    """
    import torch
    from torch_geometric.data import Data

    filepath = Path(filepath)
    config = TARGET_CONFIGS[target]
    remap = remap_override or config["remap"]
    void_label = config["void_label"]

    # --- Read point cloud via PDAL ---
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [{"type": "readers.las", "filename": str(filepath)}]
    }))
    pipeline.execute()
    points = pipeline.arrays[0]
    dim_names = list(points.dtype.names)
    n = len(points)

    # --- Position ---
    pos = np.column_stack([
        points["X"].astype(np.float32),
        points["Y"].astype(np.float32),
        points["Z"].astype(np.float32),
    ])

    # --- RGB (normalise uint16 [0, 65535] → float [0, 1]) ---
    if "Red" in dim_names and "Green" in dim_names and "Blue" in dim_names:
        r = points["Red"].astype(np.float32)
        g = points["Green"].astype(np.float32)
        b = points["Blue"].astype(np.float32)

        # Detect range and normalise
        max_val = max(r.max(), g.max(), b.max())
        if max_val > 255:
            # uint16 range (either [0, 65535] or [0, 65280])
            rgb = np.column_stack([r, g, b]) / 65535.0
        elif max_val > 1:
            # uint8 range [0, 255]
            rgb = np.column_stack([r, g, b]) / 255.0
        else:
            # Already normalised
            rgb = np.column_stack([r, g, b])
    else:
        log.warning("  No RGB channels found — setting to zeros")
        rgb = np.zeros((n, 3), dtype=np.float32)

    # --- Intensity (normalise to [0, 1]) ---
    if "Intensity" in dim_names:
        intensity_raw = points["Intensity"].astype(np.float32)
        i_max = intensity_raw.max()
        if i_max > 0:
            intensity = (intensity_raw / i_max).reshape(-1, 1)
        else:
            intensity = np.zeros((n, 1), dtype=np.float32)
    else:
        log.warning("  No Intensity channel found — setting to zeros")
        intensity = np.zeros((n, 1), dtype=np.float32)

    # --- Labels ---
    if "Classification" in dim_names:
        raw_labels = points["Classification"]
        y = remap_labels(raw_labels, remap, void_label)
    else:
        log.warning("  No Classification — all points labelled as void")
        y = np.full(n, void_label, dtype=np.int64)

    # --- Build Data object ---
    data = Data(
        pos=torch.from_numpy(pos),
        rgb=torch.from_numpy(rgb.astype(np.float32)),
        intensity=torch.from_numpy(intensity.astype(np.float32)),
        y=torch.from_numpy(y),
    )

    return data


# ===========================================================================
# PLY Export (for SPT standard pipeline)
# ===========================================================================

def write_spt_ply(
    filepath: str | Path,
    pos: np.ndarray,
    rgb: np.ndarray,
    intensity: np.ndarray,
    y: np.ndarray,
) -> None:
    """Write an SPT-compatible PLY file.

    SPT's standard datasets read PLY files.  This function writes the
    point cloud in a format that SPT's preprocessing pipeline can ingest.

    The PLY has columns: x, y, z, red, green, blue, intensity, label
    where RGB are float [0,1] and intensity is float [0,1].
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        # Fallback: write a simple ASCII PLY
        _write_ply_ascii(filepath, pos, rgb, intensity, y)
        return

    n = len(pos)
    vertex = np.zeros(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "f4"), ("green", "f4"), ("blue", "f4"),
        ("intensity", "f4"),
        ("scalar_Classification", "i4"),
    ])
    vertex["x"] = pos[:, 0]
    vertex["y"] = pos[:, 1]
    vertex["z"] = pos[:, 2]
    vertex["red"] = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"] = rgb[:, 2]
    vertex["intensity"] = intensity.ravel()
    vertex["scalar_Classification"] = y.astype(np.int32)

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(str(filepath))


def _write_ply_ascii(filepath, pos, rgb, intensity, y):
    """Fallback ASCII PLY writer (no plyfile dependency)."""
    n = len(pos)
    filepath = Path(filepath)
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float red\n")
        f.write("property float green\n")
        f.write("property float blue\n")
        f.write("property float intensity\n")
        f.write("property int scalar_Classification\n")
        f.write("end_header\n")

        for i in range(n):
            f.write(
                f"{pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f} "
                f"{rgb[i, 0]:.6f} {rgb[i, 1]:.6f} {rgb[i, 2]:.6f} "
                f"{intensity.ravel()[i]:.6f} {int(y[i])}\n"
            )


# ===========================================================================
# Diagnostics
# ===========================================================================

def diagnose_file(filepath: Path, target: str = "dales") -> dict:
    """Inspect a LAS/LAZ file and report SPT-relevant statistics."""
    filepath = Path(filepath)
    config = TARGET_CONFIGS[target]

    log.info("Diagnosing: %s (target=%s)", filepath.name, target)

    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [{"type": "readers.las", "filename": str(filepath)}]
    }))
    pipeline.execute()
    points = pipeline.arrays[0]
    dim_names = list(points.dtype.names)
    n = len(points)

    diag = {
        "input": str(filepath),
        "target": target,
        "point_count": n,
        "dimensions": dim_names,
    }

    # RGB range
    for ch in ["Red", "Green", "Blue"]:
        if ch in dim_names:
            arr = points[ch]
            diag[f"{ch}_range"] = f"[{arr.min()}, {arr.max()}]"

    # Intensity range
    if "Intensity" in dim_names:
        arr = points["Intensity"]
        diag["intensity_range"] = f"[{arr.min()}, {arr.max()}]"

    # Classification distribution + remapped
    if "Classification" in dim_names:
        raw = points["Classification"]
        remapped = remap_labels(raw, config["remap"], config["void_label"])

        unique_raw, counts_raw = np.unique(raw, return_counts=True)
        diag["asprs_distribution"] = {
            int(c): int(n) for c, n in zip(unique_raw, counts_raw)
        }

        unique_remap, counts_remap = np.unique(remapped, return_counts=True)
        diag["remapped_distribution"] = {}
        for code, count in zip(unique_remap, counts_remap):
            name = config["classes"].get(int(code), f"code_{code}")
            pct = count / n * 100
            diag["remapped_distribution"][f"{int(code)}_{name}"] = {
                "count": int(count), "percent": round(pct, 1),
            }
            log.info("    %d (%s): %d pts (%.1f%%)", int(code), name,
                     int(count), pct)

    return diag


# ===========================================================================
# File Processing
# ===========================================================================

def process_file(
    input_path: Path,
    output_dir: Path,
    target: str = "dales",
    output_format: str = "ply",
    diagnose_only: bool = False,
) -> dict:
    """Convert a single LAS/LAZ file to SPT-compatible format.

    Args:
        input_path:   Path to input LAS/LAZ.
        output_dir:   Output directory.
        target:       "dales" or "kitti360".
        output_format: "ply" (SPT standard) or "npy" (numpy arrays).
        diagnose_only: Only inspect, don't convert.

    Returns:
        Dict with processing metadata.
    """
    input_path = Path(input_path)
    log.info("Processing: %s → target=%s", input_path.name, target)
    t0 = time.time()

    if diagnose_only:
        return diagnose_file(input_path, target)

    config = TARGET_CONFIGS[target]

    # Read via PDAL
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [{"type": "readers.las", "filename": str(input_path)}]
    }))
    pipeline.execute()
    points = pipeline.arrays[0]
    dim_names = list(points.dtype.names)
    n = len(points)

    # Position
    pos = np.column_stack([
        points["X"].astype(np.float32),
        points["Y"].astype(np.float32),
        points["Z"].astype(np.float32),
    ])

    # RGB → [0, 1]
    if "Red" in dim_names:
        r = points["Red"].astype(np.float32)
        g = points["Green"].astype(np.float32)
        b = points["Blue"].astype(np.float32)
        max_val = max(r.max(), g.max(), b.max())
        divisor = 65535.0 if max_val > 255 else (255.0 if max_val > 1 else 1.0)
        rgb = np.column_stack([r, g, b]) / divisor
    else:
        rgb = np.zeros((n, 3), dtype=np.float32)

    # Intensity → [0, 1]
    if "Intensity" in dim_names:
        raw_i = points["Intensity"].astype(np.float32)
        i_max = raw_i.max()
        intensity = (raw_i / i_max).reshape(-1, 1) if i_max > 0 else np.zeros((n, 1), dtype=np.float32)
    else:
        intensity = np.zeros((n, 1), dtype=np.float32)

    # Labels
    if "Classification" in dim_names:
        y = remap_labels(points["Classification"], config["remap"], config["void_label"])
    else:
        y = np.full(n, config["void_label"], dtype=np.int64)

    # Write
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "ply":
        out_path = output_dir / f"{input_path.stem}.ply"
        write_spt_ply(out_path, pos, rgb, intensity, y)
    elif output_format == "npy":
        out_path = output_dir / f"{input_path.stem}.npz"
        np.savez_compressed(
            out_path, pos=pos, rgb=rgb, intensity=intensity, y=y,
        )

    elapsed = time.time() - t0
    log.info("  ✓ Written %s (%d pts, %.1f s)", out_path.name, n, elapsed)

    # Log remapped distribution
    unique, counts = np.unique(y, return_counts=True)
    log.info("  Label distribution (%s):", target)
    for code, count in zip(unique, counts):
        name = config["classes"].get(int(code), f"code_{code}")
        log.info("    %d (%s): %d pts (%.1f%%)",
                 int(code), name, int(count), count / n * 100)

    return {
        "input": str(input_path),
        "output": str(out_path),
        "target": target,
        "point_count": n,
        "elapsed_seconds": round(elapsed, 2),
    }


# ===========================================================================
# Batch Processing
# ===========================================================================

def process_batch(
    input_dir: Path,
    output_dir: Path,
    target: str = "dales",
    output_format: str = "ply",
    diagnose_only: bool = False,
) -> list[dict]:
    """Convert all LAS/LAZ files in a directory."""
    input_dir = Path(input_dir)
    laz = sorted(input_dir.glob("*.laz"))
    las = sorted(input_dir.glob("*.las"))
    all_files = sorted(set(laz + las), key=lambda p: p.name)

    log.info("Found %d files in %s", len(all_files), input_dir)

    results = []
    for i, fp in enumerate(all_files, 1):
        log.info("--- %d/%d ---", i, len(all_files))
        try:
            results.append(process_file(
                fp, output_dir, target, output_format, diagnose_only,
            ))
        except Exception as e:
            log.error("Failed: %s — %s", fp.name, e)
            results.append({"input": str(fp), "error": str(e)})

    ok = sum(1 for r in results if "error" not in r)
    log.info("Complete: %d/%d files", ok, len(all_files))
    return results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="prepare_spt",
        description=(
            "Convert colourised LAS/LAZ point clouds to SPT-compatible "
            "PLY format for DALES or KITTI-360 checkpoint inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Target Datasets:
  dales    — 8 classes (ground, vegetation, cars, trucks, power lines,
             fences, poles, buildings) + void. Best for ALS data.
  kitti360 — 19 classes (road, sidewalk, building, wall, fence, pole,
             etc.) + void. Best for MLS/MMS data.

Examples:
  # Convert for DALES checkpoint
  python prepare_spt.py \\
      --input-dir ./colourised/ \\
      --output-dir ./spt_data/raw/test/ \\
      --target dales

  # Convert for KITTI-360 checkpoint
  python prepare_spt.py \\
      --input-dir ./colourised/ \\
      --output-dir ./spt_data/raw/test/ \\
      --target kitti360

  # Diagnose — inspect label mapping
  python prepare_spt.py \\
      --input-dir ./colourised/ \\
      --target dales \\
      --diagnose-only

Python API (for SPT integration):
  from prepare_spt import read_las_cloud
  data = read_las_cloud("tile.laz", target="dales")
  # Returns torch_geometric.data.Data with:
  #   data.pos       — (N, 3) float32
  #   data.rgb       — (N, 3) float32 [0, 1]
  #   data.intensity — (N, 1) float32 [0, 1]
  #   data.y         — (N,) int64

Output Structure:
  Place output PLY files into the SPT data directory:
    data/<dataset>/raw/test/<tile_name>.ply
  Then run SPT preprocessing + evaluation as normal.
""",
    )

    ig = parser.add_mutually_exclusive_group(required=True)
    ig.add_argument("--input", "-i", type=str)
    ig.add_argument("--input-dir", type=str)

    parser.add_argument("--output-dir", "-o", type=str, default="./spt_data/raw/test/")
    parser.add_argument(
        "--target", "-t", type=str, default="dales",
        choices=["dales", "kitti360"],
        help="Target dataset checkpoint (default: dales).",
    )
    parser.add_argument(
        "--format", type=str, default="ply",
        choices=["ply", "npy"],
        help="Output format (default: ply for SPT standard pipeline).",
    )
    parser.add_argument("--diagnose-only", "-d", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json-summary", type=str, default="")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = []
    if args.input:
        results.append(process_file(
            Path(args.input), Path(args.output_dir),
            args.target, args.format, args.diagnose_only,
        ))
    elif args.input_dir:
        results = process_batch(
            Path(args.input_dir), Path(args.output_dir),
            args.target, args.format, args.diagnose_only,
        )

    if args.json_summary:
        p = Path(args.json_summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
