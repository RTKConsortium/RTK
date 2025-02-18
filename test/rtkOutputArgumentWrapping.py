import itk
from itk import RTK as rtk

# Test for rtkConvexShape::IsIntersectedByRay

q = rtk.QuadricShape.New()

# Define a sphere of radius 10
radius = 10
q.SetA(1)
q.SetB(1)
q.SetC(1)
q.SetJ(-radius**2)

print(q.IsIntersectedByRay([-20, 0, 0], [1, 0, 0]))

# Test for rtk::FieldOfViewImageFilter::ComputeFOVRadius

geometry = rtk.ThreeDCircularProjectionGeometry.New()
numberOfProjections = 360
firstAngle = 0
angularArc = 360
sid = 600 # source to isocenter distance in mm
sdd = 1200 # source to detector distance in mm
isox = 0 # X coordinate on the projection image of isocenter
isoy = 0 # Y coordinate on the projection image of isocenter
for x in range(0,numberOfProjections):
  angle = firstAngle + x * angularArc / numberOfProjections
  geometry.AddProjection(sid,sdd,angle,isox,isoy)

constantImageSource = rtk.ConstantImageSource[itk.Image[itk.F,3]].New()
origin = [ -127.5, -127.5, 0. ]
sizeOutput = [ 256, 256,  numberOfProjections ]
spacing = [ 1.0, 1.0, 1.0 ]
constantImageSource.SetOrigin( origin )
constantImageSource.SetSpacing( spacing )
constantImageSource.SetSize( sizeOutput )
constantImageSource.SetConstant(0.0)
source = constantImageSource.GetOutput()

f = rtk.FieldOfViewImageFilter.New()
f.SetGeometry(geometry)
f.SetProjectionsStack(source)
print(f.ComputeFOVRadius(f.FOVRadiusType_RADIUSINF))
