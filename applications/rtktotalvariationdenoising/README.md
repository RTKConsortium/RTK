# Total Variation Denoising

![noisy-fdk](../../documentation/docs/ExternalData/Noisy-fdk.png){w=200px alt="Noisy fdk reconstruction"}
![denoised-fdk](../../documentation/docs/ExternalData/Denoised-fdk.png){w=200px alt="Denoised fdk reconstruction"}

Performs total variation denoising on a 3D image using Bregman iterations.

```{literalinclude} TotalVariationDenoising.sh
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtktotalvariationdenoising/rtktotalvariationdenoising.py
	:func: build_parser
	:nodescription:
```
::::
