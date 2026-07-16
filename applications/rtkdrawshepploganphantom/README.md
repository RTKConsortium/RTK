# Draw shepp-logan phantom

Computes a 3D voxelized Shepp & Logan phantom with noise according to [https://www.slaney.org/pct/pct-errata.html]

See rtkfdk example:

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
