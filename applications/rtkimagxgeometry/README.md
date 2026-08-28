# Imagx geometry

Reader for ImagX-style detector geometries. Produces an ITK-compatible `geometry.xml` from ImagX exports (headers + images).

The command below uses these public RTK files:

- [calibration.xml](https://data.kitware.com/api/v1/item/5b179c998d777f15ebe20212/download)
- [room.xml](https://data.kitware.com/api/v1/item/5b179ca08d777f15ebe2021b/download)
- [1.dcm](https://data.kitware.com/api/v1/item/5b179c968d777f15ebe2020f/download)

`rtkimagxgeometry` needs a calibration file, a room setup file, and a stack of projections.


```bash
rtkimagxgeometry \
  --calibration calibration.xml \
  --room_setup room.xml \
  -p . \
  -r '1\\.dcm' \
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
