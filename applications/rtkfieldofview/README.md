# Field of view

Computes the field of view of a reconstruction using the acquisition geometry and the projections stack.

By default this uses RTK's fast `FieldOfViewImageFilter` (suitable for standard circular/cylindrical scans). For non-cylindrical or irregular acquisition geometries, `--bp` forces a more robust but slower backprojection-based method: the program backprojects a stack of projections filled with ones and thresholds the result to derive the FOV footprint. `--bp` requires `--mask` and is not compatible with `--displaced`.

```bash
# Basic FOV mask (fast, preferred)
rtkfieldofview -g geometry.xml -p /projections -r '.*.mha' --reconstruction recon.mha -o fov_mask.mha --mask

# Backprojection-based FOV (non-cylindrical geometry)
rtkfieldofview -g geometry.xml -p /projections -r '.*.mha' --reconstruction recon.mha -o fov_bp.mha --bp --mask --hardware cuda
```

## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtkfieldofview/rtkfieldofview.py
  :func: build_parser
  :nodescription:
```
::::
