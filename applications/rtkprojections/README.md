# Projections

Reads raw projection images, converts them to attenuation and stacks them into a single output image file.
This application is primarily used to apply preprocessing to input projections (attenuation conversion, adding noise, cropping, filtering, etc.).

```bash
# Stack projections from the current folder into a single file
rtkprojections -o projections.mha -p . -r '.*.his'

# Add additive Gaussian noise while stacking (standard deviation)
rtkprojections -o projections_noisy.mha -p . -r '.*.his' --gaussian 0.01

# Simulate pre-log Poisson noise (I0) and then additive Gaussian (optional)
rtkprojections -o projections_noisy2.mha -p . -r '.*.his' --poisson 1e5 0.01879 --gaussian 0.01
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkprojections/rtkprojections.py
	:func: build_parser
	:nodescription:
```
::::
