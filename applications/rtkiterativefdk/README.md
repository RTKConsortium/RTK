# Iterative FDK

Iterative Feldkamp-Davis-Kress reconstruction (Iterative FDK). This application performs an iterative correction of an FDK reconstruction using scaled forward-projected residuals. It supports CPU and CUDA backends and the common forward projector options; `--subsetsize` controls how many projections are processed at a time.

`````{tab-set}

````{tab-item} 3D

## 3D

This script uses the SheppLogan phantom.

```{literalinclude} IterativeFDK3D.sh
```

````
````{tab-item} 2D

## 2D

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.

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
