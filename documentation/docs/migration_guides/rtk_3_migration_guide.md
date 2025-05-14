# RTK v3 Migration Guide

## Deprecated argument: `--dimension` → Use `--size` instead

As of RTK 3.0, the `--dimension` argument in applications is deprecated and will be removed in a future release.
Please use the `--size` argument instead.

This update aligns RTK with upstream ITK's member name, where `--size` more accurately reflects the expected parameter (e.g., image size) and avoids confusion with spatial dimensions (e.g., 2D vs 3D).