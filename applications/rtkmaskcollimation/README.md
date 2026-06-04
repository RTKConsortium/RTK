# Mask collimation

Masks out the collimator from projection images using the acquisition geometry. Useful to remove detector edges and collimator shadows before downstream processing.

```bash
rtkmaskcollimation -g geometry.xml -p . -r '.*.mha' -o corrected_projections.mha
```
