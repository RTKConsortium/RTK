# Bioscan geometry

Geometry reader for the Bioscan NanoSPECT/CT format. Generates an RTK-compatible geometry XML file from Bioscan header exports and associated x-ray projection stack.

The command below uses this public RTK projection file: [bioscan.dcm](https://data.kitware.com/api/v1/item/5b179c728d777f15ebe201e2/download)

The Bioscan geometry is stored in the DICOM projections header, so `rtkbioscangeometry` retrieves it from it.

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
