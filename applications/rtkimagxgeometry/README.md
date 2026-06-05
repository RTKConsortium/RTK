# iMagX geometry

Reader for iMagX-style detector geometries. Produces an ITK-compatible `geometry.xml` from an iMagX projection stack and optional calibration and room-setup files.

The command below uses these public RTK files:

- [calibration.xml](https://data.kitware.com/api/v1/item/5b179c998d777f15ebe20212/download)
- [room.xml](https://data.kitware.com/api/v1/item/5b179ca08d777f15ebe2021b/download)
- [1.dcm](https://data.kitware.com/api/v1/item/5b179c968d777f15ebe2020f/download)

`rtkimagxgeometry` requires a stack of projections. For iMagX 1.2/1.5, provide both the calibration and room setup files to use them, or omit both to read the geometry from the projections. iMagX 2.1 requires the calibration file.


```bash
rtkimagxgeometry \
  --calibration calibration.xml \
  --room_setup room.xml \
  -p . \
  -r '1\.dcm' \
  -o geometry.xml
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtkimagxgeometry/rtkimagxgeometry.py
  :func: build_parser
  :nodescription:
```
::::
