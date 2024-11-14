# How to Define a Phantom

This guide provides instructions for defining a phantom’s geometry, including volume and surface geometries. Each geometry type and configuration is specified in square brackets:

[type : commands]

The **type** represents the geometry shape (e.g., sphere, box), while **commands** specify the parameters of the shape. By default, all values are zero.

## Volume Geometries

Volume geometries are the building blocks of phantom objects. Here’s how to define various types of volume geometries and apply optional clip planes for custom shapes.

### Volume Geometry Types

- **Sphere**: Defined by radius `r`.
- **Box**: Defined by edge lengths `dx`, `dy`, `dz`, aligned with the coordinate axes.
- **Cylinder_x, Cylinder_y, Cylinder_z**: Cylinders aligned with the x, y, or z-axis, with length `l` and radius `r`.
- **Cylinder**: A cylinder with arbitrary orientation, where `axis(expression, expression, expression)` defines its direction.
- **Ellipsoid**: Defined by half-axes `dx`, `dy`, `dz`, aligned with coordinate axes.
- **Ellipsoid_free**: Triaxial ellipsoid with arbitrary axis directions using vectors `a_x`, `a_y`, and `a_z` to define orientation.
- **Ellipt_Cyl**: Cylinder with elliptical cross-section and arbitrary orientation, defined by length `l` and half-axes `dx`, `dy`.
- **Cone**: Truncated cone with arbitrary orientation, specified by lengths `l`, and radii `r1`, `r2` at each end.
- **Tetrahedron**: Defined by four corner points `p1`, `p2`, `p3`, and `p4`.

### Using Clip Planes

All volume geometries can be customized with **clip planes** to trim parts of the shape. Clip planes define which parts of the geometry are included based on specific conditions:

- **Inequality Format**: For example, `x < expression` or `y > expression` to include only points that meet the condition.
- **Arbitrary Plane**: Defined by a normal vector `r(expression, expression, expression)` and its distance to the origin.

Multiple clip planes can be combined for complex shape intersections.

### Example Volume Geometry Definitions

Here are a few examples of defining volume geometries:

**Without Clip Planes**:
- `[Sphere: r=4]`: Creates a sphere with radius 4 cm centered at the origin.
- `[Box: x=1 y=1 z=2 dx=2 dy=2 dz=4]`: Creates a box centered at `(1,1,2)` with dimensions 2x2x4 cm.
- `[Cylinder: l=10 r=2 axis(1,1,1)]`: Creates a cylinder of length 10 cm and radius 2 cm, pointing in the `(1,1,1)` direction.

**With Clip Planes**:
- `[Sphere: r=5 x<0 y<0]`: Creates a quarter sphere in the negative x and y regions.
- `[Box: x=0.5 y=0.5 z=0.5 dx=1 dy=1 dz=1 r(1,1,1)<1/sqrt(3)]`: Creates a box truncated to form a tetrahedron.

## Surface Geometries

Surface geometries define detectors, antiscatter grids, and similar elements. The surface reference point is given by `x`, `y`, `z` or a vector `center(expression, expression, expression)`.

### Types of Surface Geometries

- **Plane_xy, Plane_xz, Plane_yz**: Rectangular planes aligned with the xy, xz, or yz planes.
- **Plane**: An arbitrary rectangular plane defined by two orthogonal vectors: `norm` for surface normal, and `a_x`, `a_y` for row and column directions.
- **Cylindrical_z**: Cylindrical detector surface parallel to the z-axis.
- **Cylindrical**: Cylindrical detector with an arbitrary axis direction.
- **Spherical**: Spherical detector geometry.

Using these volume and surface geometries, you can construct complex, realistic phantoms for tomography and imaging applications.
