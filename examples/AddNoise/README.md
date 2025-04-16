# Add Poisson noise

AddNoise illustrates how one can add pre-log Poisson noise to RTK-simulated projections. This simulation has been used in previous articles, e.g.[\[Rit et al, Med Phys, 2016\]](https://doi.org/10.1118/1.4945418) and described as *The same simulations were repeated with Poisson noise. The Shepp Logan densities were weighted by 0.01879 mm$^{−1}$, i.e., the linear attenuation coefficient of water at 75 keV. The number of photons received per detector pixel without object in the beam was constant (...) and equal to $10^4$.*
`````{tab-set}

````{tab-item} C++

```{literalinclude} ./AddNoise.cxx
:language: c++
```
````

````{tab-item} Python

```{literalinclude} ./AddNoise.py
:language: python
```

````
`````
The plot resulting from the Python version is

![AddNoise](../../documentation/docs/ExternalData/AddNoise.png){w=800px alt="Profile"}
