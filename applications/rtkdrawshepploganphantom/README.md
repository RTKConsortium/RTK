# Draw Shepp-Logan phantom

Computes a 3D voxelized Shepp & Logan phantom, with optional Gaussian noise, using the corrected parameters from the [Kak and Slaney errata](https://www.slaney.org/pct/pct-errata.html).

See the [rtkfdk example](../rtkfdk/README.md):

```{literalinclude} ../rtkfdk/FDK3D.sh
```


## Command line options


 ::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
  :filename: applications/rtkdrawshepploganphantom/rtkdrawshepploganphantom.py
  :func: build_parser
  :nodescription:
```
::::
