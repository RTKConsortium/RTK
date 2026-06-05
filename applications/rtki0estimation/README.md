# I0 estimation

Estimate I0 from projection images using an RLS-based estimator.

```bash
rtki0estimation -p . -r '.*.mha' -l 0.85 -e 10000 --debug i0_estimates.csv
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtki0estimation/rtki0estimation.py
	:func: build_parser
	:nodescription:
```
::::
