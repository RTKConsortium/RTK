# Overlay phase and shroud

Generates an RGB image showing phase minima (the descending resets of the phase signal) overlaid on an Amsterdam shroud image. The output is a standard RGB image (PNG/JPG) that highlights the detected minima in red.

This example uses a set of [projections](https://data.kitware.com/api/v1/item/5be99af88d777f2179a2e144/download) of the POPI patient from the [amsterdamshroud example](../rtkamsterdamshroud/README.md).

```{literalinclude} OverlayPhaseAndShroud.sh
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkoverlayphaseandshroud/rtkoverlayphaseandshroud.py
	:func: build_parser
	:nodescription:
```
::::
