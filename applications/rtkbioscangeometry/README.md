# Bioscan geometry

Reader for Bioscan-format detector geometries. Generates an ITK-compatible geometry XML file from Bioscan header exports and associated projection image stacks.

The command below uses this public RTK projection file: [bioscan.dcm](https://data.kitware.com/api/v1/item/5b179c728d777f15ebe201e2/download)

The Bioscan geometry is stored in the DICOM projection headers, so `rtkbioscangeometry` reads the projection files directly.

```bash
rtkbioscangeometry \
	--output bioscan_geometry.xml \
	--path . \
	--regexp 'bioscan.dcm' \
	--verbose
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkbioscangeometry/rtkbioscangeometry.py
	:func: build_parser
	:nodescription:
```
::::
