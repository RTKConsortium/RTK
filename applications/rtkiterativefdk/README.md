# Iterative FDK

Iterative Feldkamp-Davis-Kress reconstruction (Iterative FDK). This application performs an iterative correction of an FDK reconstruction using projection-subset based updates. It supports CPU and CUDA backends and common projector options.

`````{tab-set}

````{tab-item} 3D

## 3D

![sin_3D](../../documentation/docs/ExternalData/SheppLogan-Sinogram-3D.png){w=200px alt="SheppLogan sinogram 3D "}
![img_3D](../../documentation/docs/ExternalData/IterativeFdk-3D.png){w=200px alt="Iterative FDK reconstruction 3D"}

This script uses the SheppLogan phantom.

```{literalinclude} IterativeFDK3D.sh
```

````
````{tab-item} 2D

## 2D

![sin_2D](../../documentation/docs/ExternalData/SheppLogan-Sinogram-2D.png){w=200px alt="SheppLogan sinogram 2D"}
![img_2D](../../documentation/docs/ExternalData/IterativeFdk-2D.png){w=200px alt="Iterative FDK reconstruction 2D"}

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
