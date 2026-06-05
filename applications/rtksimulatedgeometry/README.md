# Simulated geometry

![geom](../../documentation/docs/ExternalData/SimulatedGeometry.png){w=400px alt="Simulated geometry"}

Creates an RTK geometry file for a simulated, regular acquisition trajectory.

Instead of converting a vendor-specific format, this application generates a geometry from a set of parameters: number of projections, angular arc, source and detector distances, and the usual gantry angles. It is the quickest way to obtain a geometry for a synthetic dataset, e.g. after drawing and forward projecting a phantom.

See the [geometry documentation](../../documentation/docs/Geometry.md) for more information on RTK geometry files and on the meaning of each parameter.

```bash
# 180 projections over a full turn (defaults: sdd 1536 mm, sid 1000 mm)
rtksimulatedgeometry -o geometry.xml -n 180
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtksimulatedgeometry/rtksimulatedgeometry.py
  :func: build_parser
  :nodescription:
```
::::
