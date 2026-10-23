# Total nuclear variation denoising

![img](../../documentation/docs/ExternalData/TotalNuclearVariationDenoising.png){w=800 alt="Total nuclear variation denoising"}

Denoises a 3D image with several channels (e.g. spectral or multi-material volumes) using total nuclear variation.

Unlike [total variation denoising](../rtktotalvariationdenoising/README.md), which treats every channel independently, total nuclear variation couples the channels through the nuclear norm of the gradient. It therefore preserves structures that are correlated across channels while suppressing noise in each of them, which is what makes it suited to material or energy decomposition results.

The `--gamma` parameter weights the regularization term, and `--niter` sets the number of iterations of the solver. Increasing `--gamma` denoises more aggressively; a value of `0` disables the regularization.

```bash
# Denoise a multi-material volume with the default settings (gamma 1.0, 5 iterations)
rtktotalnuclearvariationdenoising -i input.mha -o denoised.mha

# Stronger regularization over more iterations
rtktotalnuclearvariationdenoising -i input.mha -o denoised.mha -g 2.5 -n 20
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtktotalnuclearvariationdenoising/rtktotalnuclearvariationdenoising.py
  :func: build_parser
  :nodescription:
```
::::
