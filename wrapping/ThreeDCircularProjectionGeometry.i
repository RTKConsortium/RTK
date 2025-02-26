%include "typemaps.i"

// rtk::ThreeDCircularProjectionGeometry::FixAngles
%apply double &OUTPUT {double & outOfPlaneAngleRAD, double & gantryAngleRAD, double & inPlaneAngleRAD};