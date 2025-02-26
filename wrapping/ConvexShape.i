%include "typemaps.i"

// rtkConvexShape::IsIntersectedByRay
%apply double &OUTPUT {double & nearDist, double & farDist};