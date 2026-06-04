# Digisens geometry

Geometry reader for the Digisens file format. Generates an RTK geometry XML file from Digisens calibration files.

The command below uses this public RTK file: [calibration.cal](https://data.kitware.com/api/v1/item/5b179c768d777f15ebe201e6/download)

```bash
rtkdigisensgeometry \
  --xml_file calibration.cal \
  -o geometry.xml
```
