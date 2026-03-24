# Exporting predictions to LAZ

This guide covers how to export NAG data and inference results as
compressed LAZ point cloud files.

## Prerequisites

Install laspy with laszip support:

```bash
pip install laspy[laszip]
```

## Exporting from a notebook

After running inference and attaching predictions to the NAG:

```python
# Compute voxel-wise semantic predictions
nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)

# Export to LAZ
from src.utils.export_laz import data_to_laz

data_to_laz(nag[0], "output.laz")
```

The function automatically picks up `pos`, `rgb`, `intensity`, and
`semantic_pred` from the data object.

### Including extra attributes

You can write additional scalar attributes as extra LAZ dimensions:

```python
data_to_laz(nag[0], "output.laz", extra_attrs=["logits"])
```

### Exporting a different hierarchy level

```python
from src.utils.export_laz import nag_to_laz

# Level 0 = voxel-level (default) — exports the full dense point cloud
nag_to_laz(nag, "output.laz", level=0)

# Level 1 = superpoint-level — exports one point per superpoint (centroids)
nag_to_laz(nag, "superpoints.laz", level=1)
```

#### Understanding hierarchy levels

SPT organises point clouds into a nested hierarchy (NAG):

- **Level 0** contains the individual voxel-level points — the full dense
  point cloud.
- **Level 1** contains **superpoints** — clusters of level-0 points. Each
  superpoint is represented by a single centroid (the mean position of all
  points in the cluster). This means a level-1 export will look sparse: if
  level 0 has 1 million points, level 1 might only have ~10 000 centroids.
- Higher levels (2, 3, …) continue the hierarchy with progressively fewer,
  coarser clusters.

**If you want the full point cloud tagged with superpoint IDs**, export
level 0 and include `super_index` as an extra attribute:

```python
nag_to_laz(nag, "output.laz", level=0, extra_attrs=["super_index"])
```

This writes every dense point with a `super_index` scalar dimension that
identifies which superpoint each point belongs to.

## Exporting from saved HDF5 prediction files

The model's `track_batch()` method saves NAG objects (with predictions)
to HDF5 files during validation or testing. To enable this, set the
following in your model config:

```yaml
# Save all test batches
model:
  track_test_idx: -1

# Or save specific validation batches periodically
model:
  track_val_idx: -1
  track_val_every_n_epoch: 1
```

Files are saved to `<logger_save_dir>/predictions/<stage>/<epoch>/batch_<idx>.h5`.

Then convert to LAZ:

```python
from src.data import NAG
from src.utils.export_laz import nag_to_laz

nag = NAG.load("predictions/test/0/batch_0.h5")
nag_to_laz(nag, "output.laz")
```

## Command-line usage

```bash
python -m src.utils.export_laz predictions/test/0/batch_0.h5 output.laz

# Specify hierarchy level
python -m src.utils.export_laz input.h5 output.laz --level 1

# Include extra attributes
python -m src.utils.export_laz input.h5 output.laz --extra-attrs logits
```

## What gets exported

| SPT attribute    | LAZ field                       |
|------------------|---------------------------------|
| `pos` (XYZ)     | X, Y, Z coordinates            |
| `rgb` [0,1]     | Red, Green, Blue (uint16)       |
| `intensity` [0,1] | Intensity (uint16)            |
| `semantic_pred`  | Classification (uint8)          |
| `y` (ground truth) | Extra dimension "ground_truth" |
