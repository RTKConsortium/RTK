# Cupping correction

This example illustrates how to apply empirical cupping correction using the [algorithm of Kachelriess et al.](https://doi.org/10.1118/1.2188076) named [WaterPrecorrection](https://www.openrtk.org/Doxygen/classrtk_1_1WaterPrecorrectionImageFilter.html) in RTK. The example uses a Gate simulation using the [fixed forced detection actor](https://opengate.readthedocs.io/en/latest/tools_to_interact_with_the_simulation_actors.html#fixed-forced-detection-ct).

The simulation implements a 120 kV beam, a detector with 512x3 pixels and an energy response curve. Only the primary beam is simulated.

The simulation files, the output projections and the processing script are available [here](https://data.kitware.com/api/v1/file/5d394cea877dfcc9022c922b/download).

```{literalinclude} WaterPreCorrection.py
```
The resulting central profiles are

![Profile](Profile.png){w=800px alt="Profile"}
