# Create gammex phantom

![img](../../documentation/docs/ExternalData/GammexPhantom.png){w=400px alt="Gammex"}

This script uses the file [Gammex.txt](https://data.kitware.com/api/v1/file/6762da8a290777363f95c293/download) as configuration file which creates a Gammex phantom.

```
 # Create a 3D Gammex phantom
 rtkdrawgeometricphantom --phantomfile Gammex.txt -o gammex.mha
 ```