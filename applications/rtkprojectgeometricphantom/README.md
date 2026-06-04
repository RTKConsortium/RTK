# Project geometric phantom

Computes projections through a user-defined geometric phantom according to a given acquisition geometry. See the [Phantom documentation](../../documentation/docs/Phantom.md)

```bash
# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections of the phantom file
rtkprojectgeometricphantom -g geometry.xml -o projections.mha --spacing 2 --size=512,512,512 --phantomfile phantom.txt --phantomscale=256,256,256
```
