# Mask collimation

Masks out the region behind the collimator jaws from projection images using the acquisition geometry. Useful to remove collimator shadows before downstream processing.

```bash
rtkmaskcollimation -g geometry.xml -p . -r '.*.mha' -o corrected_projections.mha
```
