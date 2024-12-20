# How To Define a Phantom

This guide provides instructions for defining a phantom’s object according to the Forbild format.
Each shape type and configuration is specified in square brackets:

[type : commands]

The **type** represents the shape (e.g., sphere, box), while **commands** specify the parameters of the shape.
By default, all values are zero unless explicitly set.

## General Format

- Each shape is defined in a block enclosed in `{}`.
- Blocks must include the density value of the material (`rho`).
- Clip planes, unions, and other constraints can be applied to shapes.

**Example**:
```
{
    [Sphere: x=0 y=0 z=0 r=5]
    rho = 1.0
}
```

## Volumes

Volumes are the building blocks of phantom objects. These represent solid shapes with parameters that define their size, orientation, and position. All parameters are initialized to zero.

### Volumes Types

1. **Sphere**
- A perfect sphere defined by its center and radius.
- Parameters: center (`x`, `y`, `z`), radius (`r`).
- **Example**:
    ```
    {
        [Sphere: x=0 y=0 z=0 r=100]
        rho = 1.0
    }
    ```

2. **Box**
- A rectangular cuboid defined by its center and edge lengths.
- Parameters: center (`x`, `y`, `z`), edge lengths (`dx`, `dy`, `dz`).
- **Example**:
    ```
    {
        [Box: x=0 y=0 z=0 dx=100 dy=100 dz=100]
        rho = 1.5
    }
    ```

3. **Cylinder_x, Cylinder_y, Cylinder_z**
- Cylinders aligned along the x, y, or z axis.
- Parameters: center (`x`, `y`, `z`), radius (`r`), length (`l`).
- **Example**:
    ```
    {
        [Cylinder_y: x=0 y=0 z=0 r=50 l=200]
        rho = 2.0
    }
    ```

4. **Cylinder**
- Arbitrary-axis cylinder defined by an axis vector, radius, and length.
- Parameters: center (`x`, `y`, `z`), radius (`r`), length (`l`), axis vector (`axis`).
- **Example**:
    ```
    {
        [Cylinder: x=0 y=0 z=0 r=50 l=200 axis(1,1,0)]
        rho = 2.5
    }
    ```

5. **Ellipsoid**
- An ellipsoid defined by its center and half-axes lengths.
- Parameters: center (`x`, `y`, `z`), half-axes lengths (`dx`, `dy`, `dz`).
- **Example**:
    ```
    {
        [Ellipsoid: x=0 y=0 z=0 dx=100 dy=75 dz=50]
        rho = 3.0
    }
    ```

6. **Ellipsoid_free**
- An ellipsoid with arbitrary orientation.
- Parameters: center (`x`, `y`, `z`), half-axes lengths (`dx`, `dy`, `dz`), orientation vectors (`a_x`, `a_y`,`a_z`).
- **Example**:
    ```
    {
        [Ellipsoid_free: x=0 y=0 z=0 dx=100 dy=75 dz=50 a_x(1,0,0) a_y(0,1,0) a_z(0,0,1)]
        rho = 1.8
    }
    ```

7. **Ellipt_Cyl**
- A cylinder with an elliptical cross-section and arbitrary orientation.
- Parameters: center (`x`, `y`, `z`), half-axes (`dx`, `dy`, `dz`), length (`l`), orientation vectors (`a_x`, `a_y`), and principal direction (`axis`).
- **Example**:
    ```
    {
        [Ellipt_Cyl: x=0 y=0 z=0 l=150 dx=75 dy=50 dz=30 axis(0,1,1) a_x(1,0,0) a_y(0,1,0)]
        rho = 3.0
    }
    ```

8. **Cone**
- A truncated cone with varying radii at each end and arbitrary orientation.
- Parameters: center (`x`, `y`, `z`), length (`l`), radii at the base and tip (`r1`, `r2`), axis vector (`axis`).
- **Example**:
    ```
    {
        [Cone: x=0 y=0 z=0 l=150 r1=75 r2=30 axis(0,0,1)]
        rho = 2.2
    }
    ```

9. **Cone_x, Cone_y, Cone_z**
- A cone aligned along the x, y, or z-axis.
- Parameters: center of the base (`x`, `y`, `z`), length along the axis (`l`), radii at the base and tip (`r1`, `r2`).
- **Example**:
    ```
    {
        [Cone_y: x=1 y=1 z=1 l=120 r1=40 r2=0]
        rho = 2.2
    }
    ```

## Clip Planes

Clip planes can be applied to any volume to restrict their extents.
- Coordinate-based: `x<value`, `y>value`.
- Plane-based: `r(x,y,z)<value`.

**Example**:
```
{
    [Sphere: x=0 y=0 z=0 r=100 x<50 z>0]
    rho = 1.0
}
```

## Union

The `union` parameter allows combining two shapes into a single object.
- It takes a negative integer value (`union=-N`), representing an index offset between the current shape and the shape it is being united with.
- The `union` parameter is valid only when the two shapes share the same density (`rho`).
- The shapes will be combined into a single volume without gaps or overlaps.

**Example**:
```
{
    [Sphere: x=100 y=0 z=0 r=50]
    rho = 1.0
}
{
    [Box: x=0 y=0 z=0 dx=100 dy=100 dz=100]
    rho = 1.0
    union=-1
}
```

## Handling Overlapping Shapes

When adding a new shape, the system evaluates the voxel at the center of the new shape to determine the density difference. This difference is used to calculate how much value is added to the phantom. If a previous shape already occupies the center voxel and has the same density as the new shape, the added density will be `0`. As a result, the new shape may not appear as expected.

If you face this problem, try defining the new shape before the previous one if the center of the previous shape does not fall inside the new shape’s form.
