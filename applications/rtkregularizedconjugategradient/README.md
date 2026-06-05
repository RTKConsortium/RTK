# Regularized Conjugate Gradient

Alternates between conjugate gradient reconstruction and spatial regularization (total variation and wavelets) to produce constrained reconstructions.

```{literalinclude} RegularizedConjugateGradient.sh

```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkregularizedconjugategradient/rtkregularizedconjugategradient.py
	:func: build_parser
	:nodescription:
```
::::
