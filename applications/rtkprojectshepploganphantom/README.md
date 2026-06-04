# Project Shepp-Logan Phantom

![sin](../../documentation/docs/ExternalData/SheppLogan-Sinogram-3D.png){w=200px alt="SheppLogan sinogram 3D"}

Computes projections through a 3D Shepp & Logan phantom according to a geometry file.

```bash
# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections with a 3D geometry
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 200

# Use the reconstruction filter of your choice like rtkfdk or rtkconjugategradient
```
