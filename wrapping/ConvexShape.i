%include "typemaps.i"

// rtkConvexShape::IsIntersectedByRay
%apply double &OUTPUT {double & infDist, double & supDist};
