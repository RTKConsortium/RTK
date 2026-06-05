# Lag correction

Fourth-order LTI lag correction for projection stacks.

```bash
# Replace the -a (rates) and -c (coefficients) values with calibrated detector parameters
rtklagcorrection -p . -r projections.mha -o corrected_projections.mha \
  -a 0.0 0.1 0.05 0.01 \
  -c 1.0 0.5 0.2 0.1
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtklagcorrection/rtklagcorrection.py
  :func: build_parser
  :nodescription:
```
::::
