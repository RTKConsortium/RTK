/*=========================================================================
 *
 *  Copyright Insight Software Consortium & RTK Consortium
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
#ifndef __srtkPixelIDValues_h
#define __srtkPixelIDValues_h

#include "srtkCommon.h"
#include "srtkPixelIDTypeLists.h"

#include <string>
#include <ostream>

namespace rtk
{
namespace simple
{

typedef int PixelIDValueType;

template < typename TPixelID >
struct PixelIDToPixelIDValue
{
  enum { Result = typelist::IndexOf<InstantiatedPixelIDTypeList, TPixelID >::Result };
};

template <typename TImageType>
struct ImageTypeToPixelIDValue
{
  enum { Result = PixelIDToPixelIDValue< typename ImageTypeToPixelID<TImageType>::PixelIDType>::Result };
};

/** \brief Enumerated values of pixelIDs
 *
 * Each PixelID's value correspondes to the index of the PixelID type,
 * in the type list "InstantiatedPixelIDTypeList". It is possible that
 * different configurations for SimpleRTK could result in different
 * values for pixelID. So these enumerated values should be used.
 *
 * Additionally, not all PixelID an instantiated in for the tool
 * kit. If a PixelID is not instantiated then it's value is
 * -1. Therefore it is likely that multiple elements in the
 * enumeration will have a zero value. Therefore the first prefered
 * methods is to use "if" statements, with the first branch checking
 * for the Unknown value.
 *
 * If a switch case statement is needed the ConditionalValue
 * meta-programming object can be used as follows:
 * \code
 *  switch( pixelIDValue )
 *     {
 *   case srtk::srtkUnknown:
 *     // handle exceptional case
 *     break
 *   case srtk::ConditionalValue< srtk::srtkUInt8 != srtk::srtkUnknown, srtk::srtkUInt8, -2 >::Value:
 *     ...
 *     break;
 *   case srtk::ConditionalValue< srtk::srtkInt8 != srtk::srtkUnknown, srtk::srtkInt8, -3 >::Value:
 *     ...
 *     break;
 *   case srtk::ConditionalValue< srtk::N != srtk::srtkUnknown, srtk::N, -N >::Value:
 *     ...
 *     break;
 *   default:
 *      // handle another exceptoinal case
 *     }
 * \endcode
 */
enum PixelIDValueEnum {
  srtkUnknown = -1,
  srtkUInt8 = PixelIDToPixelIDValue< BasicPixelID<uint8_t> >::Result,   //< Unsigned 8 bit integer
  srtkInt8 = PixelIDToPixelIDValue< BasicPixelID<int8_t> >::Result,     //< Signed 8 bit integer
  srtkUInt16 = PixelIDToPixelIDValue< BasicPixelID<uint16_t> >::Result, //< Unsigned 16 bit integer
  srtkInt16 = PixelIDToPixelIDValue< BasicPixelID<int16_t> >::Result,   //< Signed 16 bit integer
  srtkUInt32 = PixelIDToPixelIDValue< BasicPixelID<uint32_t> >::Result, //< Unsigned 32 bit integer
  srtkInt32 = PixelIDToPixelIDValue< BasicPixelID<int32_t> >::Result,   //< Signed 32 bit integer
  srtkUInt64 = PixelIDToPixelIDValue< BasicPixelID<uint64_t> >::Result, //< Unsigned 64 bit integer
  srtkInt64 = PixelIDToPixelIDValue< BasicPixelID<int64_t> >::Result,   //< Signed 64 bit integer
  srtkFloat32 = PixelIDToPixelIDValue< BasicPixelID<float> >::Result,   //< 32 bit float
  srtkFloat64 = PixelIDToPixelIDValue< BasicPixelID<double> >::Result,  //< 64 bit float
  srtkComplexFloat32 = PixelIDToPixelIDValue< BasicPixelID<std::complex<float> > >::Result,  //< compelex number of 32 bit float
  srtkComplexFloat64 = PixelIDToPixelIDValue< BasicPixelID<std::complex<double> > >::Result,  //< compelex number of 64 bit float
  srtkVectorUInt8 = PixelIDToPixelIDValue< VectorPixelID<uint8_t> >::Result, //< Multi-component of unsigned 8 bit integer
  srtkVectorInt8 = PixelIDToPixelIDValue< VectorPixelID<int8_t> >::Result, //< Multi-component of signed 8 bit integer
  srtkVectorUInt16 = PixelIDToPixelIDValue< VectorPixelID<uint16_t> >::Result, //< Multi-component of unsigned 16 bit integer
  srtkVectorInt16 = PixelIDToPixelIDValue< VectorPixelID<int16_t> >::Result, //< Multi-component of signed 16 bit integer
  srtkVectorUInt32 = PixelIDToPixelIDValue< VectorPixelID<uint32_t> >::Result, //< Multi-component of unsigned 32 bit integer
  srtkVectorInt32 = PixelIDToPixelIDValue< VectorPixelID<int32_t> >::Result, //< Multi-component of signed 32 bit integer
  srtkVectorUInt64 = PixelIDToPixelIDValue< VectorPixelID<uint64_t> >::Result, //< Multi-component of unsigned 64 bit integer
  srtkVectorInt64 = PixelIDToPixelIDValue< VectorPixelID<int64_t> >::Result, //< Multi-component of signed 64 bit integer
  srtkVectorFloat32 = PixelIDToPixelIDValue< VectorPixelID<float> >::Result, //< Multi-component of 32 bit float
  srtkVectorFloat64 = PixelIDToPixelIDValue< VectorPixelID<double> >::Result,  //< Multi-component of 64 bit float
  srtkLabelUInt8 = PixelIDToPixelIDValue< LabelPixelID<uint8_t> >::Result, //< RLE label of unsigned 8 bit integers
  srtkLabelUInt16 = PixelIDToPixelIDValue< LabelPixelID<uint16_t> >::Result, //< RLE label of unsigned 16 bit integers
  srtkLabelUInt32 = PixelIDToPixelIDValue< LabelPixelID<uint32_t> >::Result, //< RLE label of unsigned 32 bit integers
  srtkLabelUInt64 = PixelIDToPixelIDValue< LabelPixelID<uint64_t> >::Result, //< RLE label of unsigned 64 bit integers
};



const std::string SRTKCommon_EXPORT GetPixelIDValueAsString( PixelIDValueType type );
const std::string SRTKCommon_EXPORT GetPixelIDValueAsString( PixelIDValueEnum type );

#ifndef SWIG
SRTKCommon_EXPORT std::ostream& operator<<(std::ostream& os, const PixelIDValueEnum id);
#endif



}
}
#endif // _srtkPixelIDValues_h
