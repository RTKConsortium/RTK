# Scatter Glare Correction

Corrects projection images for scatter glare using a deconvolution kernel.

```bash
# Apply glare correction using two kernel coefficients
rtkscatterglarecorrection -p . -r '.*.his' -c 0.8 0.2 -o projections_corrected.mha --verbose

# Output the difference image
rtkscatterglarecorrection -p . -r '.*.his' -c 0.8 0.2 -d -o diff.mha
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkscatterglarecorrection/rtkscatterglarecorrection.py
	:func: build_parser
	:nodescription:
```
::::
