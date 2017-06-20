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


#include <Rdefines.h>
#include <Rversion.h>

#include "srtkImage.h"
#include "srtkConditional.h"
#include "srtkImportImageFilter.h"

SEXP ImAsArray(rtk::simple::Image src)
{
  // tricky to make this efficient with memory and fast.
  // Ideally we want multithreaded casting directly to the
  // R array. We could use a Cast filter and then a memory copy,
  // obviously producing a redundant copy. If we do a direct cast,
  // then we're probably not multi-threaded.
  // Lets be slow but memory efficient.

  std::vector<unsigned int> sz = src.GetSize();
  rtk::simple::PixelIDValueType  PID=src.GetPixelIDValue();
  SEXP res = 0;
  double *dans=0;
  int *ians=0;
  unsigned pixcount=src.GetNumberOfComponentsPerPixel();
  for (unsigned k = 0; k < sz.size();k++)
    {
    pixcount *= sz[k];
    }
  switch (PID)
    {
    case rtk::simple::srtkUnknown:
    {
    char error_msg[1024];
    snprintf( error_msg, 1024, "Exception thrown ImAsArray : unkown pixel type");
    Rprintf(error_msg);
    return(res);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkInt8, -3 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt8, -15 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt8, -2 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt8, -14 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkInt16, -5 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt16, -17 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt16, -4 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt16, -16 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkInt32, -7 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt32, -19 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt32, -6 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt32, -18 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt64, -8 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt64, -20 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkInt64, -9 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt64, -21 >::Value:
    {
    // allocate an integer array
    PROTECT(res = Rf_allocVector(INTSXP, pixcount));
    ians = INTEGER_POINTER(res);
    }
    break;
    default:
    {
    // allocate double array
    PROTECT(res = Rf_allocVector(REALSXP, pixcount));
    dans = NUMERIC_POINTER(res);
    }
    }

  switch (PID)
    {
    case rtk::simple::ConditionalValue< rtk::simple::srtkUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt8, -2 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt8, -14 >::Value:
    {
    uint8_t * buff = src.GetBufferAsUInt8();
    std::copy(buff,buff + pixcount,ians);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkInt8, -3 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt8 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt8, -15 >::Value:
    {
    int8_t * buff = src.GetBufferAsInt8();
    std::copy(buff,buff + pixcount,ians);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt16, -4 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt16, -16 >::Value:
    {
    uint16_t * buff = src.GetBufferAsUInt16();
    std::copy(buff,buff + pixcount,ians);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkInt16, -5 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt16 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt16, -17 >::Value:
    {
    int16_t * buff = src.GetBufferAsInt16();
    std::copy(buff,buff + pixcount,ians);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt32, -6 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt32, -18 >::Value:
    {
    uint32_t * buff = src.GetBufferAsUInt32();
    std::copy(buff,buff + pixcount,ians);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkInt32, -7 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt32, -19 >::Value:
    {
    int32_t * buff = src.GetBufferAsInt32();
    std::copy(buff,buff + pixcount,ians);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkUInt64, -8 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorUInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorUInt64, -20 >::Value:

    {
    uint64_t * buff = src.GetBufferAsUInt64();
    std::copy(buff,buff + pixcount,ians);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkInt64, -9 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorInt64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorInt64, -21 >::Value:
    {
    int64_t * buff = src.GetBufferAsInt64();
    std::copy(buff,buff + pixcount,ians);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkFloat32 != rtk::simple::srtkUnknown, rtk::simple::srtkFloat32, -10 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorFloat32 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorFloat32, -22 >::Value:
    {
    float * buff = src.GetBufferAsFloat();
    std::copy(buff,buff + pixcount,dans);
    }
    break;
    case rtk::simple::ConditionalValue< rtk::simple::srtkFloat64 != rtk::simple::srtkUnknown, rtk::simple::srtkFloat64, -11 >::Value:
    case rtk::simple::ConditionalValue< rtk::simple::srtkVectorFloat64 != rtk::simple::srtkUnknown, rtk::simple::srtkVectorFloat64, -23 >::Value:
    {
    double * buff = src.GetBufferAsDouble();
    std::copy(buff,buff + pixcount,dans);
    }
    break;
    default:
      char error_msg[1024];
      snprintf( error_msg, 1024, "Exception thrown ImAsArray : unsupported pixel type: %d", PID );
      Rprintf(error_msg);
    }
  UNPROTECT(1);
  return(res);
}

rtk::simple::Image ArrayAsIm(SEXP arr,
                             std::vector<unsigned int> size,
                             std::vector<double> spacing,
                             std::vector<double> origin)
{
  // can't work out how to get the array size in C
  rtk::simple::ImportImageFilter importer;
  importer.SetSpacing( spacing );
  importer.SetOrigin( origin );
  importer.SetSize( size );
  if (Rf_isReal(arr))
    {
    importer.SetBufferAsDouble(NUMERIC_POINTER(arr));
    }
  else if (Rf_isInteger(arr) || Rf_isLogical(arr))
    {
    importer.SetBufferAsInt32(INTEGER_POINTER(arr));
    }
  else
    {
    char error_msg[1024];
    snprintf( error_msg, 1024, "Exception thrown ArrayAsIm : unsupported array type");
    Rprintf(error_msg);
    }
  rtk::simple::Image res = importer.Execute();
  return(res);
}
