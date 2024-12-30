# 3D Projection Geometry

## Purpose

The purpose of this page is to describe the geometry object used in RTK to relate a tomography to projection images. There is currently only one geometry class, `ThreeDCircularProjectionGeometry`.

## Units

- Degrees are used in the geometry files and to pass angle values to the `AddProjection` member but they are converted to and stored in radians. Angles are wrapped between 0 and 360 degrees.
- No unit is enforced for distances but it is the responsibility of the user to have a consistent unit for all distances (pixel and voxel spacings, geometry parameters...). Millimeters are typically used in ITK and DICOM.

## Image Coordinate System

An `itk::Image<>` contains information to convert voxel indices to physical coordinates using its members `m_Origin`, `m_Spacing`, and `m_Direction`. Voxel coordinates are not used in RTK except for internal computation. The conversion from voxel index coordinates to physical coordinates and the dimensions of the images are out of the scope of this document (see section 4.1.1 of the [ITK software guide](https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch4.html)). In the following, origin refers to point with coordinates $\vec{0}$ mm (and not to `m_Origin` in ITK).

## Fixed Coordinate System

The fixed coordinate system $(x,y,z)$ in RTK is the coordinate system of the tomography with the isocenter at the origin $(0,0,0)$.

## ProjectionGeometry\<TDimension\>

This is the mother class for relating a TDimension-D tomography to a (TDimension-1)-D projection image. It holds a vector of (TDimension)x(TDimension+1) projection matrices accessible via `GetMatrices`. The construction of those matrices is geometry dependent.

## ThreeDCircularProjectionGeometry

This class is meant to define a set of 2D projection images, acquired with a flat panel along a circular trajectory, around a 3D tomography. The trajectory does not have to be strictly circular but it is assumed in some reconstruction algorithms that the rotation axis is y. The description of the geometry is based on the international standard IEC 61217 which has been designed for cone-beam imagers on isocentric radiotherapy systems but it can be used for any 3D circular trajectory. The fixed coordinate system of RTK and the fixed coordinate system of IEC 61217 are the same.

9 parameters are used per projection to define the position of the source and the detector relatively to the fixed coordinate system. The 9 parameters are set with the method `AddProjection`. Default values are provided for the parameters which are not mandatory. Note that explicit names have been used but this does not necessarily correspond to the value returned by the scanner which can use its own parameterization.

### Detector Orientation

#### Initial Detector Orientation

With all parameters set to 0, the detector is normal to the z direction of the fixed coordinate system, similarly to the x-ray image receptor in the IEC 61217.

#### Rotation Order

Three rotation angles are used to define the orientation of the detector. The ZXY convention of Euler angles is used for detector orientation where `GantryAngle` is the rotation around y, `OutOfPlaneAngle` the rotation around x, and `InPlaneAngle` the rotation around z. These three angles are detailed in the following.

#### GantryAngle

Gantry angle of the scanner. It corresponds to $\phi_g$ in Section 2.3 of IEC 61217:

> The rotation of the "g" system is defined by the rotation of coordinate axes Xg, Zg by an angle $\phi_g$ about axis Yg (therefore about Yf of the "f" system).
>
> An increase in the value of $\phi_g$ corresponds to a clockwise rotation of the GANTRY as viewed along the horizontal axis Yf from the ISOCENTER towards the GANTRY.

#### OutOfPlaneAngle

Out of plane rotation of the flat panel complementary to the GantryAngle rotation, i.e., with a rotation axis perpendicular to the gantry rotation axis and parallel to the flat panel. It is optional with a default value equal to 0. There is no corresponding rotation in IEC 61217. After gantry rotation, the rotation is defined by the rotation of the coordinate axes y and z about x. An increase in the value of `OutOfPlaneAngle` corresponds to a counter-clockwise rotation of the flat panel as viewed from a positive value along the x axis towards the isocenter.

#### InPlaneAngle

In plane rotation of the 2D projection. It is optional with 0 as default value. If `OutOfPlaneAngle` equals 0, it corresponds to $\theta_r$ in Section 2.6 of IEC 61217:

> The rotation of the "r" system is defined by the rotation of the coordinate axes Xr, Yr about Zr (parallel to axis Zg) by an angle $\theta_r$.
>
> An increase in the value of angle $\theta_r$ corresponds to a counter-clockwise rotation of the X-RAY IMAGE RECEPTOR as viewed from the RADIATION SOURCE.

#### Rotation Matrix

The rotation matrix in homogeneous coordinates is then (constructed with `itk::Euler3DTransform<double>::ComputeMatrix()` with opposite angles because we rotate the volume coordinates instead of the scanner):

$$
\begin{split}
M_R = &
\begin{pmatrix}
\cos(-InPlaneAngle) & -\sin(-InPlaneAngle) & 0 & 0\\
\sin(-InPlaneAngle) & \cos(-InPlaneAngle) & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{pmatrix}\\
&\times
\begin{pmatrix}
1 & 0 & 0 & 0\\
0 & \cos(-OutOfPlaneAngle) & -\sin(-OutOfPlaneAngle) & 0\\
0 & \sin(-OutOfPlaneAngle) & \cos(-OutOfPlaneAngle) & 0\\
0 & 0 & 0 & 1
\end{pmatrix}\\
&\times
\begin{pmatrix}
\cos(-GantryAngle) & 0 & \sin(-GantryAngle) & 0 \\
0 & 1 & 0 & 0 \\
-\sin(-GantryAngle) & 0 & \cos(-GantryAngle) & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
\end{split}
$$

### Drawings

The following drawing describes the parameters of the source and the detector positions in the rotated coordinate system $(Rx,Ry,Rz)$ (i.e., oriented according to the detector orientation), with its origin at the isocenter, when all values are positive (but all distances can be negative in this geometry):

![Drawing](https://www.openrtk.org/RTK/img/ThreeDCircularProjectionGeometry.svg)

These 6 parameters are used to describe any source and detector positions. It is simpler to understand the circular geometry when all Offset values equal 0:

![Drawing](https://www.openrtk.org/RTK/img/ThreeDCircularProjectionGeometry_aligned.svg)

### Source Position

The source position is defined with respect to the isocenter with three parameters, `SourceOffsetX`, `SourceOffsetY`, and `SourceToIsocenterDistance`. `(SourceOffsetX, SourceOffsetY, SourceToIsocenterDistance)` are the coordinates of the source in the rotated coordinated system. In IEC 61217, `SourceToIsocenterDistance` is the RADIATION SOURCE axis distance, SAD. `SourceOffsetX` and `SourceOffsetY` are optional and zero by default.

### Detector Position

The detector position is defined with respect to the source with three parameters: `ProjectionOffsetX`, `ProjectionOffsetY`, and `SourceToDetectorDistance`. `(ProjectionOffsetX, ProjectionOffsetY, SourceToIsocenterDistance - SourceToDetectorDistance)` are the coordinates of the detector origin $(0,0)$ in the rotated coordinated system. In IEC 61217, `SourceToDetectorDistance` is the RADIATION SOURCE to IMAGE RECEPTION AREA distance, SID. `ProjectionOffsetX` and `ProjectionOffsetY` are optional and zero by default.

### Final Matrix

Each matrix, accessible via `GetMatrices`, is constructed with:

$$
 \begin{split}
  M_P =
  &\begin{pmatrix}
    1 & 0 & SourceOffsetX-ProjectionOffsetX  \\
    0 & 1 & SourceOffsetY-ProjectionOffsetY  \\
    0 & 0 & 1
  \end{pmatrix}\\
  &\times
  \begin{pmatrix}
    -SourceToDetectorDistance & 0 & 0 & 0  \\
    0 & -SourceToDetectorDistance & 0 & 0  \\
    0 & 0 & 1 & -SourceToIsocenterDistance
  \end{pmatrix}\\
  &\times
  \begin{pmatrix}
    1 & 0 & 0 & -SourceOffsetX  \\
    0 & 1 & 0 & -SourceOffsetY  \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1
  \end{pmatrix}\\
  &\times
  M_R
  \end{split}
$$

### Detector Radius

In addition to flat panel detectors, some of the forward and back projectors in
RTK can handle cylindrical detectors. The radius of the cylindrical detector is
stored only once, as the variable `RadiusCylindricalDetector`. The default value
for `RadiusCylindricalDetector` is 0, and indicates that the detector is a flat
panel (i.e. infinite radius, but 0 is easier to deal with). When the value is
non-zero, then the flat detector is curved according to the radius and remain
tangent to the corresponding flat detector along the line defined by the
detector origin `(0,0)` and second axis of the detector without accouting
for the parameters `ProjectionOffsetX` and `ProjectionOffsetY`. The latter two
allow to modify the Origin of each projection as is the case for a flat panel.
The cylindrical detector geometry is illustrated in the following scheme:

![Drawing](https://www.openrtk.org/RTK/img/ThreeDCircularProjectionGeometry_cylindrical.svg)

This scheme is based on the previous one with all offsets equal 0 but this is not required.

### Parallel Geometry

When `SourceToDetectorDistance` is set to 0, the geometry is assumed to be
parallel (i.e. infinite distance, but 0 is easier to deal with). The detector
is then flat. The rays are perpendicular to the detector plane which is
oriented similarly to the divergent geometry.  The (plane) source is actually
placed at a distance `SourceToIsocenterDistance` from the isocenter and the
detector is placed symetrically around the origin `(0,0,0)` at the same
`SourceToIsocenterDistance`. This is summarized in the following scheme:

![Drawing](https://www.openrtk.org/RTK/img/ThreeDCircularProjectionGeometry_parallel.svg)

In this case, the projection matrix becomes:

$$
  M_P =
  \begin{pmatrix}
    1 & 0 & 0 & -ProjectionOffsetX  \\
    0 & 1 & 0 & -ProjectionOffsetY  \\
    0 & 0 & 0 & 1
  \end{pmatrix}
  \times M_R.
$$

### XML File

`ThreeDCircularProjectionGeometry` can be saved and loaded from an XML file
with
[rtk::ThreeDCircularProjectionGeometryXMLFileReader](http://www.openrtk.org/Doxygen/classrtk_1_1ThreeDCircularProjectionGeometryXMLFileReader.html)
and
[rtk::ThreeDCircularProjectionGeometryXMLFileWriter](http://www.openrtk.org/Doxygen/classrtk_1_1ThreeDCircularProjectionGeometryXMLFileWriter.html),
respectivey. If the parameter is equal to the default value for all
projections, it is not stored in the file. If it is equal for all projections
but different from the default value, it is stored once. Otherwise, it is
stored for each projection.  The matrix is given for information. It is read
and checked to be consistent with the parameters but a manual modification of
the file must consistently modify both the parameters and the matrix. An
example is given hereafter:

```xml
<?xml version="1.0"?>
<!DOCTYPE RTKGEOMETRY>
<RTKThreeDCircularGeometry version="3">
  <SourceToIsocenterDistance>1000</SourceToIsocenterDistance>
  <SourceToDetectorDistance>1536</SourceToDetectorDistance>
  <RadiusCylindricalDetector>1536</RadiusCylindricalDetector>
  <Projection>
    <GantryAngle>271.847274780273</GantryAngle>
    <ProjectionOffsetX>-117.056503295898</ProjectionOffsetX>
    <ProjectionOffsetY>-1.01195001602173</ProjectionOffsetY>
    <Matrix>
          -166.5093078829                   0   -1531.42837748039   -117056.503295898
        -1.01142410874151               -1536  0.0326206557691505   -1011.95001602173
       -0.999480303105996                   0  0.0322354417240802               -1000
    </Matrix>
  </Projection>
  <Projection>
    <GantryAngle>271.852905273438</GantryAngle>
    <ProjectionOffsetX>-117.056831359863</ProjectionOffsetX>
    <ProjectionOffsetY>-1.01187002658844</ProjectionOffsetY>
    <Matrix>
        -166.660129424325                   0   -1531.41199650136   -117056.831359863
        -1.01134095059569               -1536  0.0327174625589984   -1011.87002658844
       -0.999477130482326                   0  0.0323336611415466               -1000
    </Matrix>
  </Projection>
</RTKThreeDCircularGeometry>
