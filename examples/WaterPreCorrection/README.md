
# WaterPreCorrection

This example illustrates how to apply empirical cupping correction using the [algorithm of Kachelriess et al.](http://onlinelibrary.wiley.com/doi/10.1118/1.2188076/abstract) named [WaterPrecorrection](http://www.openrtk.org/Doxygen/classrtk_1_1WaterPrecorrectionImageFilter.html) in RTK. The example uses a Gate simulation using the [fixed forced detection actor](http://wiki.opengatecollaboration.org/index.php/Users_Guide:Tools_to_Interact_with_the_Simulation_:_Actors#Fixed_Forced_Detection_CT).

The simulation implements a 120 kV beam, a detector with 512x3 pixels and an energy response curve. Only the primary beam is simulated.

This version uses the [Python wrapping](https://pypi.org/project/itk-rtk/). The simulation files, the output projections and the processing script are available [here](https://data.kitware.com/api/v1/file/5d394cea877dfcc9022c922b/download).

```{literalinclude} waterPreCorrection.py
```
