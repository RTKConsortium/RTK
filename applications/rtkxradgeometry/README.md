# XRad Geometry

Creates an RTK geometry XML file from an acquisition exported by the XRad system.

The public RTK test data for XRad includes these sample files:

- [SolidWater_HiGain1x1.header](https://data.kitware.com/api/v1/item/5b179cd68d777f15ebe20266/download)

`rtkxradgeometry` reads the geometry from the acquisition header.

```bash
rtkxradgeometry \
	-i SolidWater_HiGain1x1.header \
	-o geometry.xml
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtkxradgeometry/rtkxradgeometry.py
  :func: build_parser
  :nodescription:
```
::::
