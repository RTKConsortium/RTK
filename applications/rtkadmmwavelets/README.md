# Daubechies Wavelets Regularized Reconstruction

![sin](../../documentation/docs/ExternalData/SheppLogan-Sinogram-3D.png){w=300px alt="SheppLogan sinogram 3D"}
![img](../../documentation/docs/ExternalData/AdmmWavelets.png){w=300px alt="AdmmWavelets reconstruction"}

This script uses the SheppLogan phantom

```{literalinclude} DaubechiesWavelets.sh
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtkadmmwavelets/rtkadmmwavelets.py
  :func: build_parser
  :nodescription:
```
::::
