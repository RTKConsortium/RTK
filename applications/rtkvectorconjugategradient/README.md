# Vector Conjugate Gradient

Reconstructs a multi-material 3D volume from projections using a conjugate gradient method producing vector-valued voxels.

```bash
rtkvectorconjugategradient -g geometry.xml -o recon.mha --projections projections.mha --niterations 20 --verbose
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkvectorconjugategradient/rtkvectorconjugategradient.py
	:func: build_parser
	:nodescription:
```
::::
