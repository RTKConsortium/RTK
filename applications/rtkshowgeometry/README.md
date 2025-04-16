# Geometry viewer

`rtkshowgeometry` is a Python-only command line tool which provides an **interactive three-dimensional (3D) visualization** of RTK geometry files, projections and volumes using [`pyvista`](https://pyvista.org). It is useful for visually assessing a geometry, i.e., how it relates the CT image with the projection images by displaying the source positions and the projections positions and orientations in a 3D scene.

![geom](../../documentation/docs/ExternalData/ShowGeometry.png){w=800 alt="Show Geometry"}

All geometries described in the [documentation](../../documentation/docs/Geometry.md) are supported: cone-beam and parallel (`SDD = 0`), flat and cylindrical (`RadiusCylindricalDetector > 0`) detectors. If no projections are given, the detector size defaults to 40% of the Source-to-Isocenter Distance (SID) and is centered around point `(u,v)=(0,0)`.

```{literalinclude} showgeometry.sh
```
