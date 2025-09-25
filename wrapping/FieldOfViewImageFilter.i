%include "typemaps.i"

// rtk::FieldOfViewImageFilter::ComputeFOVRadius
%apply double &OUTPUT {double & x, double & z, double & r};
