import itk
from itk import RTK as rtk
import math


# Test for rtkConvexShape::IsIntersectedByRay
def test_IsIntersectedByRay():
    q = rtk.QuadricShape.New()

    # Define a sphere of radius 10
    radius = 10
    q.SetA(1)
    q.SetB(1)
    q.SetC(1)
    q.SetJ(-(radius**2))

    # Ray starting outside sphere aiming through center should intersect.
    res = q.IsIntersectedByRay([-20, 0, 0], [1, 0, 0])
    assert isinstance(res, (list, tuple)) and len(res) >= 3
    intersects, t0, t1 = res[0], res[1], res[2]
    assert intersects is True
    # Expected param distances from start point x=-20 to entry at -10 (10 units) and exit at +10 (30 units)
    assert math.isclose(t0, 10.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(t1, 30.0, rel_tol=1e-6, abs_tol=1e-6)


# Test for rtk::FieldOfViewImageFilter::ComputeFOVRadius
def test_ComputeFOVRadius():
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    numberOfProjections = 360
    firstAngle = 0
    angularArc = 360
    sid = 600  # source to isocenter distance in mm
    sdd = 1200  # source to detector distance in mm
    isox = 0  # X coordinate on the projection image of isocenter
    isoy = 0  # Y coordinate on the projection image of isocenter
    for x in range(numberOfProjections):
        angle = firstAngle + x * angularArc / numberOfProjections
        geometry.AddProjection(sid, sdd, angle, isox, isoy)

    constantImageSource = rtk.ConstantImageSource[itk.Image[itk.F, 3]].New()
    origin = [-127.5, -127.5, 0.0]
    sizeOutput = [256, 256, numberOfProjections]
    spacing = [1.0, 1.0, 1.0]
    constantImageSource.SetOrigin(origin)
    constantImageSource.SetSpacing(spacing)
    constantImageSource.SetSize(sizeOutput)
    constantImageSource.SetConstant(0.0)
    source = constantImageSource.GetOutput()

    f = rtk.FieldOfViewImageFilter.New()
    f.SetGeometry(geometry)
    f.SetProjectionsStack(source)

    res_inf = f.ComputeFOVRadius(f.FOVRadiusType_RADIUSINF)
    success = res_inf[0]
    rmin = res_inf[1]
    rmax = res_inf[2]
    rinf = res_inf[3]
    assert success is True
    # rmin and rmax are zero in this symmetric centered geometry
    assert math.isclose(rmin, 0.0, abs_tol=1e-6)
    assert math.isclose(rmax, 0.0, abs_tol=1e-6)
    # Theoretical infinite-distance FOV radius: (detector_width/2) * sid / sdd = (256/2)*600/1200 = 64
    expected = (sizeOutput[0] * spacing[0] / 2.0) * (sid / sdd)
    # Allow small deviation (<2%) due to internal computations
    rel_err = abs(rinf - expected) / expected
    assert rel_err < 0.02


if __name__ == "__main__":
    test_IsIntersectedByRay()
    test_ComputeFOVRadius()
