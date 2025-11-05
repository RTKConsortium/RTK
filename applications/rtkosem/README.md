# Ordered Subset Expectation Maximization (OSEM)

The following example illustrates the command line application `rtkosem` by reconstructing a Shepp–Logan phantom with the OSEM iterative algorithm for cone‑beam CT.

![sin_3D](../../documentation/docs/ExternalData/SheppLogan-Sinogram-3D.png){w=200px alt="SheppLogan sinogram 3D "}
![img](../../documentation/docs/ExternalData/Osem.png){w=200px alt="Osem reconstruction"}


```{literalinclude} OSEM.sh
```

For details about available forward/back projectors and their options, see the Projectors documentation: [documentation/docs/Projectors.md](../../documentation/docs/Projectors.md).
