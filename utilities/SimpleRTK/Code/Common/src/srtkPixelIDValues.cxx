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
#include "srtkPixelIDValues.h"

namespace rtk
{
namespace simple
{


const std::string GetPixelIDValueAsString( PixelIDValueEnum type )
{
  return GetPixelIDValueAsString( static_cast<PixelIDValueType>(type) );
}

const std::string GetPixelIDValueAsString( PixelIDValueType type )
{

  if ( type == srtkUnknown )
    {
    // Unknow must be first because other enums may be -1 if they are
    // not instantiated
    return "Unknown pixel id";
    }
  else if ( type == srtkUInt8 )
    {
    return "8-bit unsigned integer";
    }
  else if ( type == srtkInt8 )
    {
    return "8-bit signed integer";
    }
  else if ( type ==  srtkUInt16 )
    {
    return "16-bit unsigned integer";
    }
  else if ( type == srtkInt16 )
    {
    return "16-bit signed integer";
    }
  else if ( type == srtkUInt32 )
    {
    return "32-bit unsigned integer";
    }
  else if ( type == srtkInt32 )
    {
    return "32-bit signed integer";
    }
  else if ( type == srtkUInt64 )
    {
    return "64-bit unsigned integer";
    }
  else if ( type == srtkInt64 )
    {
    return "64-bit signed integer";
    }
  else if ( type == srtkFloat32 )
    {
    return "32-bit float";
    }
  else if ( type == srtkFloat64 )
    {
    return "64-bit float";
    }
  else if ( type == srtkComplexFloat32 )
    {
    return "complex of 32-bit float";
    }
  else if ( type == srtkComplexFloat64 )
    {
    return "complex of 64-bit float";
    }
  else if ( type == srtkVectorUInt8 )
    {
    return "vector of 8-bit unsigned integer";
    }
  else if ( type == srtkVectorInt8 )
    {
    return "vector of 8-bit signed integer";
    }
  else if ( type ==  srtkVectorUInt16 )
    {
    return "vector of 16-bit unsigned integer";
    }
  else if ( type == srtkVectorInt16 )
    {
    return "vector of 16-bit signed integer";
    }
  else if ( type == srtkVectorUInt32 )
    {
    return "vector of 32-bit unsigned integer";
    }
  else if ( type == srtkVectorInt32 )
    {
    return "vector of 32-bit signed integer";
    }
  else if ( type == srtkVectorUInt64 )
    {
    return "vector of 64-bit unsigned integer";
    }
  else if ( type == srtkVectorInt64 )
    {
    return "vector of 64-bit signed integer";
    }
  else if ( type == srtkVectorFloat32 )
    {
    return "vector of 32-bit float";
    }
  else if ( type == srtkVectorFloat64 )
    {
    return "vector of 64-bit float";
    }
  else if ( type == srtkLabelUInt8 )
    {
    return "label of 8-bit unsigned integer";
    }
  else if ( srtkLabelUInt16 )
    {
    return "label of 16-bit unsigned integer";
    }
  else if ( srtkLabelUInt32 )
    {
    return "label of 32-bit unsigned integer";
    }
  else if ( srtkLabelUInt64 )
    {
    return "label of 64-bit unsigned integer";
    }
  else
    {
    return "ERRONEOUS PIXEL ID!";
    }
}

std::ostream& operator<<(std::ostream& os, const PixelIDValueEnum id)
{
  return (os << GetPixelIDValueAsString(id));
}


}
}
