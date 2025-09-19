# Amsterdam Shroud

![sin](../../documentation/docs/ExternalData/Moving-Phantom-Sinogram.png){w=400px alt="Moving phantom sinogram"}
![img](../../documentation/docs/ExternalData/Amsterdam.png){w=400px alt="Amsterdam image"}

Picture 1 shows the sinogram of the input and picture 2 the shroud image that was created using the command line below.

The script uses the file [movingPhantom.mha](https://data.kitware.com/api/v1/file/5be99c428d777f2179a2e537/download) as input:

```
# Creating an Amsterdam Shroud image from a set of projections
rtkamsterdamshroud -p . -r movingPhantom.mha -o Amsterdam.mha
 ```
