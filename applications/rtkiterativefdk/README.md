# Iterative FDK

Iterative Feldkamp-Davis-Kress reconstruction (Iterative FDK). This application performs an iterative correction of an FDK reconstruction using scaled residuals in the projection space. It supports CPU and CUDA backends and common projector options. `--subsetsize` controls how many projections are processed at a time.

`````{tab-set}

````{tab-item} 3D

## 3D

This script uses the SheppLogan phantom.

```{literalinclude} IterativeFDK3D.sh
```

````
````{tab-item} 2D

## 2D

The same reconstruction can be performed using the original [2D Shepp-Logan phantom](https://data.kitware.com/api/v1/file/6925a4ebcdb169d7a021d788/download).

```{literalinclude} IterativeFDK2D.sh
```

````
`````

## Command line options

::::{container} argparse-no-usage
```{eval-rst}
.. argparse::
	:filename: applications/rtkiterativefdk/rtkiterativefdk.py
	:func: build_parser
	:nodescription:
```
::::
