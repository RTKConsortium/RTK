/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
%module(directors="1") SimpleRTK

// Remove some warnings
#pragma SWIG nowarn=362,503,401,389,516,511

// Use STL support
%include <std_vector.i>
%include <std_string.i>
%include <std_map.i>
#if SWIGPYTHON || SWIGRUBY
%include <std_complex.i>
#endif
// Use C99 int support
%include <stdint.i>

// Use exceptions
%include "exception.i"

// Customize exception handling
%exception {
  try {
    $action
  } catch( std::exception &ex ) {
    const size_t e_size = 10240;
    char error_msg[e_size];
// TODO this should be replaces with some try compile stuff

%#ifdef _MSC_VER
    _snprintf_s( error_msg, e_size, e_size, "Exception thrown in SimpleRTK $symname: %s", ex.what() );
%#else
    snprintf( error_msg, e_size, "Exception thrown in SimpleRTK $symname: %s", ex.what() );
%#endif

    SWIG_exception( SWIG_RuntimeError, error_msg );
  } catch( ... ) {
    SWIG_exception( SWIG_UnknownError, "Unknown exception thrown in SimpleRTK $symname" );
  }
}

// Global Tweaks to srtk::Image
%ignore rtk::simple::Image::GetITKBase( void );
%ignore rtk::simple::Image::GetITKBase( void ) const;

#ifndef SWIGCSHARP
%ignore rtk::simple::Image::GetBufferAsInt8;
%ignore rtk::simple::Image::GetBufferAsUInt8;
%ignore rtk::simple::Image::GetBufferAsInt16;
%ignore rtk::simple::Image::GetBufferAsUInt16;
%ignore rtk::simple::Image::GetBufferAsInt32;
%ignore rtk::simple::Image::GetBufferAsUInt32;
%ignore rtk::simple::Image::GetBufferAsInt64;
%ignore rtk::simple::Image::GetBufferAsUInt64;
%ignore rtk::simple::Image::GetBufferAsFloat;
%ignore rtk::simple::Image::GetBufferAsDouble;
#endif


// This section is copied verbatim into the generated source code.
// Any include files, definitions, etc. need to go here.
%{
#include <SimpleRTK.h>
#include <srtkImageOperators.h>
%}

// Language Specific Sections
%include CSharp.i
%include Java.i
%include Tcl.i
%include Python.i
%include Lua.i
%include R.i
%include Ruby.i

// Help SWIG handle std vectors
namespace std
{
  %template(VectorBool) vector<bool>;
  %template(VectorUInt8) vector<uint8_t>;
  %template(VectorInt8) vector<int8_t>;
  %template(VectorUInt16) vector<uint16_t>;
  %template(VectorInt16) vector<int16_t>;
  %template(VectorUInt32) vector<uint32_t>;
  %template(VectorInt32) vector<int32_t>;
  %template(VectorUInt64) vector<uint64_t>;
  %template(VectorInt64) vector<int64_t>;
  %template(VectorFloat) vector<float>;
  %template(VectorDouble) vector<double>;
  %template(VectorOfImage) vector< rtk::simple::Image >;
  %template(VectorUIntList) vector< vector<unsigned int> >;
  %template(VectorString) vector< std::string >;

  // Support for LabelStatisticsImageFilter/MeasurementMap
  %template(BasicMeasurementMap) map<std::string, double>;

  %template(DoubleDoubleMap) map<double, double>;
}



// define these preprocessor directives to nothing for the swig interface
#define SRTKCommon_EXPORT
#define SRTKCommon_HIDDEN
#define SRTKBasicFilters0_EXPORT
#define SRTKBasicFilters0_HIDDEN
#define SRTKBasicFilters_EXPORT
#define SRTKBasicFilters_HIDDEN
#define SRTKIO_EXPORT
#define SRTKIO_HIDDEN


// Any new classes need to have an "%include" statement to be wrapped.

// Common
%include "srtkVersion.h"
%include "srtkPixelIDValues.h"
%include "srtkImage.h"
%include "srtkTransform.h"
%include "srtkThreeDCircularProjectionGeometry.h"
%include "srtkCommand.h"
%include "srtkInterpolator.h"
//%include "srtkKernel.h"
%include "srtkEvent.h"

// Basic Filter Base
%include "srtkProcessObject.h"
%include "srtkImageFilterExecuteBase.h"
%include "srtkImageFilter.h"

// describe instantiated  templates
%template(ImageFilterExecuteBase_0) rtk::simple::ImageFilterExecuteBase<0>;
%template(ImageFilterExecuteBase_1) rtk::simple::ImageFilterExecuteBase<1>;
%template(ImageFilterExecuteBase_2) rtk::simple::ImageFilterExecuteBase<2>;
%template(ImageFilterExecuteBase_3) rtk::simple::ImageFilterExecuteBase<3>;
%template(ImageFilterExecuteBase_4) rtk::simple::ImageFilterExecuteBase<4>;
%template(ImageFilterExecuteBase_5) rtk::simple::ImageFilterExecuteBase<5>;

%template(ImageFilter_0) rtk::simple::ImageFilter<0>;
%template(ImageFilter_1) rtk::simple::ImageFilter<1>;
%template(ImageFilter_2) rtk::simple::ImageFilter<2>;
%template(ImageFilter_3) rtk::simple::ImageFilter<3>;
%template(ImageFilter_4) rtk::simple::ImageFilter<4>;
%template(ImageFilter_5) rtk::simple::ImageFilter<5>;

// IO
%include "srtkShow.h"
%include "srtkImageFileWriter.h"
%include "srtkImageReaderBase.h"
%include "srtkImageSeriesReader.h"
%include "srtkImageFileReader.h"
%include "srtkThreeDCircularProjectionGeometryXMLFileWriter.h"

// Basic Filters
%include "srtkHashImageFilter.h"
%include "srtkStatisticsImageFilter.h"
//%include "srtkMeasurementMap.h"
//%include "srtkLabelStatisticsImageFilter.h"
%include "srtkCastImageFilter.h"
//%include "srtkAdditionalProcedures.h"

// Only C# can handle import filter
#if SWIGCSHARP
%include "srtkImportImageFilter.h"
#endif


#if SWIGPYTHON
%include "srtkPyCommand.h"
#endif


// Auto-generated headers
%include "SimpleRTKBasicFiltersGeneratedHeaders.i"
