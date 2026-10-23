# Varian ProBeam geometry

![img](../../documentation/docs/ExternalData/VarianProBeam.png){w=400px alt="VarianProBeam snapshot"}

Creates an RTK geometry file from a Varian ProBeam acquisition.

The application reads the XML description of the acquisition together with the list of projection files, and converts them into an RTK geometry. The projection file names are selected with the usual input projections options (`--path` and `--regexp`).

A complete walkthrough, from downloading a ProBeam dataset to reconstructing a slice, is available in the [Varian reconstruction documentation](../rtkvarianobigeometry/README.md) under the *ProBeam* tab.

```bash
rtkvarianprobeamgeometry \
  --xml_file Scan.xml \
  --path Acquisitions/733061622 \
  --regexp Proj_.*.xim \
  -o geometry.xml
```


## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtkvarianprobeamgeometry/rtkvarianprobeamgeometry.py
  :func: build_parser
  :nodescription:
```
::::
