# Gain correction

Polynomial gain correction for projection images. Reads a set of projections, applies dark/gain calibration, and writes corrected projections.

```bash
rtkgaincorrection --calibDir Calibration/ --Gain gain.mha --Dark dark.mha -p . -r '.*.mha' -o corrected.mha
```

## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkgaincorrection/rtkgaincorrection.py
	:func: build_parser
	:nodescription:
```
::::
