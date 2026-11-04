# ORA geometry

Geometry reader for the ORA/medPhoton file format. Converts ORA export files into a `geometry.xml` used by RTK reconstruction tools.

This example dataset contains an `.xml` (metadata), an `.mhd` (header) and a `.raw` (raw data) file:
- [0_afterLog.ora.xml (metadata)](https://data.kitware.com/api/v1/item/5b179ca58d777f15ebe20222/download)
- [0_afterLog.mhd (header)](https://data.kitware.com/api/v1/item/5b179ca38d777f15ebe2021f/download)
- [0_afterLog.raw (raw data)](https://data.kitware.com/api/v1/item/5b179ca68d777f15ebe20225/download)

The `.xml` metadata references the corresponding `.mhd` and `.raw` filenames, so `rtkorageometry` only requires the `.xml` files as input.

## Generate geometry XML from an ORA XML file

Create a geometry XML from ORA-format projection files:

```bash
rtkorageometry \
  -p . \
  -r "0_afterLog.ora.xml" \
  -o geometry.xml
```

## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtkorageometry/rtkorageometry.py
  :func: build_parser
  :nodescription:
```
::::
