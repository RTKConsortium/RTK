/*=========================================================================
*
*  Copyright Insight Software Consortium
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
#include <string.h>

#include <numeric>
#include <functional>

#include "srtkImage.h"
#include "srtkConditional.h"
#include "srtkExceptionObject.h"

namespace srtk = rtk::simple;

// Python is written in C
#ifdef __cplusplus
extern "C"
{
#endif

/** An internal function that performs a deep copy of the image buffer
 * into a python byte array. The byte array can later be converted
 * into a numpy array with the from buffer method.
 */
static PyObject *
srtk_GetByteArrayFromImage( PyObject *SWIGUNUSEDPARM(self), PyObject *args )
{
  // Holds the bulk data
  PyObject * byteArray = NULL;

  const void * srtkBufferPtr;
  Py_ssize_t len;
  std::vector< unsigned int > size;
  size_t pixelSize = 1;

  unsigned int dimension;

  /* Cast over to a srtk Image. */
  PyObject * pyImage;
  void * voidImage;
  const srtk::Image * srtkImage;
  int res = 0;
  if( !PyArg_ParseTuple( args, "O", &pyImage ) )
    {
    SWIG_fail; // SWIG_fail is a macro that says goto: fail (return NULL)
    }
  res = SWIG_ConvertPtr( pyImage, &voidImage, SWIGTYPE_p_rtk__simple__Image, 0 );
  if( !SWIG_IsOK( res ) )
    {
    SWIG_exception_fail(SWIG_ArgError(res), "in method 'GetByteArrayFromImage', argument needs to be of type 'srtk::Image *'");
    }
  srtkImage = reinterpret_cast< srtk::Image * >( voidImage );

  switch( srtkImage->GetPixelIDValue() )
    {
  case srtk::srtkUnknown:
    PyErr_SetString( PyExc_RuntimeError, "Unknown pixel type." );
    SWIG_fail;
    break;
  case srtk::ConditionalValue< srtk::srtkVectorUInt8 != srtk::srtkUnknown, srtk::srtkVectorUInt8, -14 >::Value:
  case srtk::ConditionalValue< srtk::srtkUInt8 != srtk::srtkUnknown, srtk::srtkUInt8, -2 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsUInt8();
    pixelSize  = sizeof( uint8_t );
    break;
  case srtk::ConditionalValue< srtk::srtkVectorInt8 != srtk::srtkUnknown, srtk::srtkVectorInt8, -15 >::Value:
  case srtk::ConditionalValue< srtk::srtkInt8 != srtk::srtkUnknown, srtk::srtkInt8, -3 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsInt8();
    pixelSize  = sizeof( int8_t );
    break;
  case srtk::ConditionalValue< srtk::srtkVectorUInt16 != srtk::srtkUnknown, srtk::srtkVectorUInt16, -16 >::Value:
  case srtk::ConditionalValue< srtk::srtkUInt16 != srtk::srtkUnknown, srtk::srtkUInt16, -4 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsUInt16();
    pixelSize  = sizeof( uint16_t );
    break;
  case srtk::ConditionalValue< srtk::srtkVectorInt16 != srtk::srtkUnknown, srtk::srtkVectorInt16, -17 >::Value:
  case srtk::ConditionalValue< srtk::srtkInt16 != srtk::srtkUnknown, srtk::srtkInt16, -5 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsInt16();
    pixelSize  = sizeof( int16_t );
    break;
  case srtk::ConditionalValue< srtk::srtkVectorUInt32 != srtk::srtkUnknown, srtk::srtkVectorUInt32, -18 >::Value:
  case srtk::ConditionalValue< srtk::srtkUInt32 != srtk::srtkUnknown, srtk::srtkUInt32, -6 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsUInt32();
    pixelSize  = sizeof( uint32_t );
    break;
  case srtk::ConditionalValue< srtk::srtkVectorInt32 != srtk::srtkUnknown, srtk::srtkVectorInt32, -19 >::Value:
  case srtk::ConditionalValue< srtk::srtkInt32 != srtk::srtkUnknown, srtk::srtkInt32, -7 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsInt32();
    pixelSize  = sizeof( int32_t );
    break;
  case srtk::ConditionalValue< srtk::srtkVectorUInt64 != srtk::srtkUnknown, srtk::srtkVectorUInt64, -20 >::Value:
  case srtk::ConditionalValue< srtk::srtkUInt64 != srtk::srtkUnknown, srtk::srtkUInt64, -8 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsUInt64();
    pixelSize  = sizeof( uint64_t );
    break;
  case srtk::ConditionalValue< srtk::srtkVectorInt64 != srtk::srtkUnknown, srtk::srtkVectorInt64, -21 >::Value:
  case srtk::ConditionalValue< srtk::srtkInt64 != srtk::srtkUnknown, srtk::srtkInt64, -9 >::Value:
     srtkBufferPtr = (const void *)srtkImage->GetBufferAsInt64();
     pixelSize  = sizeof( int64_t );
     break;
  case srtk::ConditionalValue< srtk::srtkVectorFloat32 != srtk::srtkUnknown, srtk::srtkVectorFloat32, -22 >::Value:
  case srtk::ConditionalValue< srtk::srtkFloat32 != srtk::srtkUnknown, srtk::srtkFloat32, -10 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsFloat();
    pixelSize  = sizeof( float );
    break;
  case srtk::ConditionalValue< srtk::srtkVectorFloat64 != srtk::srtkUnknown, srtk::srtkVectorFloat64, -23 >::Value:
  case srtk::ConditionalValue< srtk::srtkFloat64 != srtk::srtkUnknown, srtk::srtkFloat64, -11 >::Value:
    srtkBufferPtr = (const void *)srtkImage->GetBufferAsDouble(); // \todo rename to Float64 for consistency
    pixelSize  = sizeof( double );
    break;
  case srtk::ConditionalValue< srtk::srtkComplexFloat32 != srtk::srtkUnknown, srtk::srtkComplexFloat32, -12 >::Value:
  case srtk::ConditionalValue< srtk::srtkComplexFloat64 != srtk::srtkUnknown, srtk::srtkComplexFloat64, -13 >::Value:
    PyErr_SetString( PyExc_RuntimeError, "Images of Complex Pixel types currently are not supported." );
    SWIG_fail;
    break;
  default:
    PyErr_SetString( PyExc_RuntimeError, "Unknown pixel type." );
    SWIG_fail;
    }

  dimension = srtkImage->GetDimension();
  size = srtkImage->GetSize();

  // if the image is a vector just treat is as another dimension
  if ( srtkImage->GetNumberOfComponentsPerPixel() > 1 )
    {
    size.push_back( srtkImage->GetNumberOfComponentsPerPixel() );
    }

  len = std::accumulate( size.begin(), size.end(), size_t(1), std::multiplies<size_t>() );
  len *= pixelSize;

  // When the string is null, the bytearray is uninitialized but allocated
  byteArray = PyByteArray_FromStringAndSize( NULL, len );
  if( !byteArray )
    {
    PyErr_SetString( PyExc_RuntimeError, "Error initializing bytearray." );
    SWIG_fail;
    }

  char *arrayView;
  if( (arrayView = PyByteArray_AsString( byteArray ) ) == NULL  )
    {
    SWIG_fail;
    }
  memcpy( arrayView, srtkBufferPtr, len );

  return byteArray;

fail:
  Py_XDECREF( byteArray );
  return NULL;
}


/** An internal function that performs a deep copy of the image buffer
 * into a python byte array. The byte array can later be converted
 * into a numpy array with the frombuffer method.
 */
static PyObject*
srtk_SetImageFromArray( PyObject *SWIGUNUSEDPARM(self), PyObject *args )
{
  PyObject * pyImage = NULL;

  const void *buffer;
  Py_ssize_t buffer_len;
  Py_buffer  pyBuffer;
  memset(&pyBuffer, 0, sizeof(Py_buffer));

  const srtk::Image * srtkImage = NULL;
  void * srtkBufferPtr = NULL;
  size_t pixelSize = 1;

  unsigned int dimension = 0;
  std::vector< unsigned int > size;
  size_t len = 1;

  // We wish to support both the new PEP3118 buffer interface and the
  // older. So we first try to parse the arguments with the new buffer
  // protocol, then the old.
  if (!PyArg_ParseTuple( args, "s*O", &pyBuffer, &pyImage ) )
    {
    PyErr_Clear();

#ifdef PY_SSIZE_T_CLEAN
    typedef Py_ssize_t bufSizeType;
#else
    typedef int bufSizeType;
#endif

    bufSizeType _len;
    // This function takes 2 arguments from python, the first is an
    // python object which support the old "ReadBuffer" interface
    if( !PyArg_ParseTuple( args, "s#O", &buffer, &_len, &pyImage ) )
      {
      return NULL;
      }
    buffer_len = _len;
    }
  else
    {
    if ( PyBuffer_IsContiguous( &pyBuffer, 'C' ) != 1 )
      {
      PyBuffer_Release( &pyBuffer );
      PyErr_SetString( PyExc_TypeError, "A C Contiguous buffer object is required." );
      return NULL;
      }
    buffer_len = pyBuffer.len;
    buffer = pyBuffer.buf;
    }

  /* Cast over to a srtk Image. */
  {
    void * voidImage;
    int res = 0;
    res = SWIG_ConvertPtr( pyImage, &voidImage, SWIGTYPE_p_rtk__simple__Image, 0 );
    if( !SWIG_IsOK( res ) )
      {
      SWIG_exception_fail(SWIG_ArgError(res), "in method 'SetImageFromArray', argument needs to be of type 'srtk::Image *'");
      }
    srtkImage = reinterpret_cast< srtk::Image * >( voidImage );
  }

  try
    {
    switch( srtkImage->GetPixelIDValue() )
      {
      case srtk::srtkUnknown:
        PyErr_SetString( PyExc_RuntimeError, "Unknown pixel type." );
        goto fail;
        break;
      case srtk::ConditionalValue< srtk::srtkVectorUInt8 != srtk::srtkUnknown, srtk::srtkVectorUInt8, -14 >::Value:
      case srtk::ConditionalValue< srtk::srtkUInt8 != srtk::srtkUnknown, srtk::srtkUInt8, -2 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsUInt8();
        pixelSize  = sizeof( uint8_t );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorInt8 != srtk::srtkUnknown, srtk::srtkVectorInt8, -15 >::Value:
      case srtk::ConditionalValue< srtk::srtkInt8 != srtk::srtkUnknown, srtk::srtkInt8, -3 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsInt8();
        pixelSize  = sizeof( int8_t );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorUInt16 != srtk::srtkUnknown, srtk::srtkVectorUInt16, -16 >::Value:
      case srtk::ConditionalValue< srtk::srtkUInt16 != srtk::srtkUnknown, srtk::srtkUInt16, -4 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsUInt16();
        pixelSize  = sizeof( uint16_t );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorInt16 != srtk::srtkUnknown, srtk::srtkVectorInt16, -17 >::Value:
      case srtk::ConditionalValue< srtk::srtkInt16 != srtk::srtkUnknown, srtk::srtkInt16, -5 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsInt16();
        pixelSize  = sizeof( int16_t );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorUInt32 != srtk::srtkUnknown, srtk::srtkVectorUInt32, -18 >::Value:
      case srtk::ConditionalValue< srtk::srtkUInt32 != srtk::srtkUnknown, srtk::srtkUInt32, -6 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsUInt32();
        pixelSize  = sizeof( uint32_t );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorInt32 != srtk::srtkUnknown, srtk::srtkVectorInt32, -19 >::Value:
      case srtk::ConditionalValue< srtk::srtkInt32 != srtk::srtkUnknown, srtk::srtkInt32, -7 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsInt32();
        pixelSize  = sizeof( int32_t );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorUInt64 != srtk::srtkUnknown, srtk::srtkVectorUInt64, -20 >::Value:
      case srtk::ConditionalValue< srtk::srtkUInt64 != srtk::srtkUnknown, srtk::srtkUInt64, -8 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsUInt64();
        pixelSize  = sizeof( uint64_t );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorInt64 != srtk::srtkUnknown, srtk::srtkVectorInt64, -21 >::Value:
      case srtk::ConditionalValue< srtk::srtkInt64 != srtk::srtkUnknown, srtk::srtkInt64, -9 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsInt64();
        pixelSize  = sizeof( int64_t );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorFloat32 != srtk::srtkUnknown, srtk::srtkVectorFloat32, -22 >::Value:
      case srtk::ConditionalValue< srtk::srtkFloat32 != srtk::srtkUnknown, srtk::srtkFloat32, -10 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsFloat();
        pixelSize  = sizeof( float );
        break;
      case srtk::ConditionalValue< srtk::srtkVectorFloat64 != srtk::srtkUnknown, srtk::srtkVectorFloat64, -23 >::Value:
      case srtk::ConditionalValue< srtk::srtkFloat64 != srtk::srtkUnknown, srtk::srtkFloat64, -11 >::Value:
        srtkBufferPtr = (void *)srtkImage->GetBufferAsDouble(); // \todo rename to Float64 for consistency
        pixelSize  = sizeof( double );
        break;
      case srtk::ConditionalValue< srtk::srtkComplexFloat32 != srtk::srtkUnknown, srtk::srtkComplexFloat32, -12 >::Value:
      case srtk::ConditionalValue< srtk::srtkComplexFloat64 != srtk::srtkUnknown, srtk::srtkComplexFloat64, -13 >::Value:
        PyErr_SetString( PyExc_RuntimeError, "Images of Complex Pixel types currently are not supported." );
        goto fail;
        break;
      default:
        PyErr_SetString( PyExc_RuntimeError, "Unknown pixel type." );
        goto fail;
      }
    }
  catch( const std::exception &e )
    {
    std::string msg = "Exception thrown in SimpleRTK new Image: ";
    msg += e.what();
    PyErr_SetString( PyExc_RuntimeError, msg.c_str() );
    goto fail;
    }


  dimension = srtkImage->GetDimension();
  size = srtkImage->GetSize();

  // if the image is a vector just treat is as another dimension
  if ( srtkImage->GetNumberOfComponentsPerPixel() > 1 )
    {
    size.push_back( srtkImage->GetNumberOfComponentsPerPixel() );
    }

  len = std::accumulate( size.begin(), size.end(), size_t(1), std::multiplies<size_t>() );
  len *= pixelSize;

  if ( buffer_len != len )
    {
    PyErr_SetString( PyExc_RuntimeError, "Size mismatch of image and Buffer." );
    goto fail;
    }

  memcpy( (void *)srtkBufferPtr, buffer, len );


  PyBuffer_Release( &pyBuffer );
  Py_RETURN_NONE;

fail:
  PyBuffer_Release( &pyBuffer );
  return NULL;
}

#ifdef __cplusplus
} // end extern "C"
#endif
