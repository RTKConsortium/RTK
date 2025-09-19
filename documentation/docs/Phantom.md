# Phantom definition

This page, adapted from the original [Forbild's syntax explanation](https://www.dkfz.de/en/roentgenbildgebung/ct/CT_Phantoms/Head_Phantom/How-to-define-a-phantom.pdf), provides instructions for defining a phantom in the Forbild format in RTK.

- Each shape is defined in a block enclosed in curly brackets `{}`.
- Each shape is specified in square brackets `[type : commands]`
with **type** in the list below (e.g., sphere, box) and **commands** the parameters of the shape.

- Each block must specify the function (attenuation, emission, etc. depending on the context) value of the material with `rho`.

**Example**:
```
{
    [Sphere: x=0 y=0 z=0 r=5]
    rho = 1.0
}
```
- Clip planes and unions (see below) are optional.

## Volumes

Volumes are the building blocks of phantom objects. These represent solid shapes with parameters that define their size, orientation and position.
**All parameter values are zero by default**.

1. **Sphere**
- Parameters: center (`x`,`y`,`z`), radius (`r`).
- **Example**:
    ```
    {
        [Sphere: x=0 y=0 z=0 r=100]
        rho = 1.0
    }
    ```

2. **Box**
- Also known as rectangular cuboid.
- Parameters: center (`x`,`y`,`z`), edge lengths (`dx`,`dy`,`dz`).
- **Example**:
    ```
    {
        [Box: x=0 y=0 z=0 dx=100 dy=100 dz=100]
        rho = 1.5
    }
    ```

3. **Cylinder_x, Cylinder_y, Cylinder_z**
- Cylinder parallel to the x, y or z axis, respectively.
- Parameters: center (`x`,`y`,`z`), radius (`r`), length (`l`).
- **Example**:
    ```
    {
        [Cylinder_y: x=0 y=0 z=0 r=50 l=200]
        rho = 2.0
    }
    ```

4. **Cylinder**
- Parameters: center (`x`,`y`,`z`), radius (`r`), length (`l`), axis vector (`axis`).
- **Example**:
    ```
    {
        [Cylinder: x=0 y=0 z=0 r=50 l=200 axis(1,1,0)]
        rho = 2.5
    }
    ```

5. **Ellipsoid**
- Parameters: center (`x`,`y`,`z`), half-axes (`dx`,`dy`,`dz`).
- **Example**:
    ```
    {
        [Ellipsoid: x=0 y=0 z=0 dx=100 dy=75 dz=50]
        rho = 3.0
    }
    ```

6. **Ellipsoid_free**
- Parameters: center (`x`,`y`,`z`), half-axes (`dx`,`dy`,`dz`), orthogonal orientation vectors (`a_x`,`a_y`,`a_z`).
- **Example**:
    ```
    {
        [Ellipsoid_free: x=0 y=0 z=0 dx=100 dy=75 dz=50 a_x(1,0,0) a_y(0,1,0) a_z(0,0,1)]
        rho = 1.8
    }
    ```

7. **Ellipt_Cyl**
- Cylinder with an elliptical cross-section.
- Parameters: center (`x`,`y`,`z`), half-axes (`dx`,`dy`,`dz`), length (`l`), orientation vectors (`a_x`,`a_y`) and principal direction (`axis`).
- **Example**:
    ```
    {
        [Ellipt_Cyl: x=0 y=0 z=0 l=150 dx=75 dy=50 dz=30 axis(0,1,1) a_x(1,0,0) a_y(0,1,0)]
        rho = 3.0
    }
    ```

8. **Cone**
- Parameters: center (`x`,`y`,`z`), length (`l`), radii at the base and tip (`r1`,`r2`), axis vector (`axis`).
- **Example**:
    ```
    {
        [Cone: x=0 y=0 z=0 l=150 r1=75 r2=30 axis(0,0,1)]
        rho = 2.2
    }
    ```

9. **Cone_x, Cone_y, Cone_z**
- Cone parallel to the x, y or z axis, respectively.
- Parameters: center of the base (`x`,`y`,`z`), length along the axis (`l`), radii at the base and tip (`r1`,`r2`).
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

## Function value `rho`

The function (attenuation, emission, etc. depending on the context) value `rho` is defined absolutely in the shape. RTK automatically deduces how much must be added to the current background based on the center (`x`,`y`,`z`) of the shape (e.g. 0.5 if one adds a shape with `rho=1.5` in a shape with `rho=1`). This is order dependent so the same phantom shapes defined in different orders might result in different results. The background of a shape must be defined first.

## Union

The `union` parameter allows taking the union of two shapes. The second shape takes a negative integer value (`union=-N`), representing an index offset between the current shape and the shape it is being united with.
The `union` parameter is valid only when the two shapes share the same density (`rho`).

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
