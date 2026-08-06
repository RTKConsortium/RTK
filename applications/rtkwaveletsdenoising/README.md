# Wavelets Denoising

Denoises a volume using Daubechies wavelets soft-thresholding.

```bash
rtkwaveletsdenoising -i noisy.mha -o denoised.mha --order 5 --level 3 --threshold 0.02 --verbose
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkwaveletsdenoising/rtkwaveletsdenoising.py
	:func: build_parser
	:nodescription:
```
::::
