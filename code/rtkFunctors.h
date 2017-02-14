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

#ifndef rtkFunctors_h
#define rtkFunctors_h

#include <itkVectorImage.h>

namespace rtk
{
namespace Functor{

/** \class LengthGetter
 * \brief Function returning the number of elements (1 for scalar, n for vector)
 *
 * \author Cyril
 *
 * \ingroup Functions
 */
template< class TImage >
class LengthGetter
{
public:
  LengthGetter() {}
  ~LengthGetter() {}

  inline unsigned int operator()(const TImage* image) const
  {
  return 1;
  }
};

template< class TImage >
class VectorLengthGetter
{
public:
  VectorLengthGetter() {}
  ~VectorLengthGetter() {}

  inline unsigned int operator()(const TImage* image) const
  {
  return (image->GetVectorLength());
  }
};


/** \class VectorLengthSetter
 * \brief Function to set the vector length of an image, if it is a vector image
 *
 * \author Cyril Mory
 *
 * \ingroup Functions
 */

template< class TImage >
class LengthSetter
{
public:
  LengthSetter() {}
  ~LengthSetter() {}

  inline void operator()( TImage* image, unsigned int vectorLength ) const {}
};

template< class TImage >
class VectorLengthSetter
{
public:
  VectorLengthSetter() {}
  ~VectorLengthSetter() {}

  inline void operator()( TImage* image, unsigned int vectorLength ) const
  {
  image->SetVectorLength(vectorLength);
  }
};

/** \class PixelFiller
 * \brief Function filling elements of the pixel (1 for scalar, n for vector)
 * with a given value
 *
 * \author Cyril
 *
 * \ingroup Functions
 */
template< class PixelValueType >
class PixelFiller
{
public:
  PixelFiller() {}
  ~PixelFiller() {}

  inline PixelValueType operator()(const PixelValueType value, unsigned int vectorLength) const
  {
  return value;
  }
};

template< class PixelValueType >
class VectorPixelFiller
{
public:
  VectorPixelFiller() {}
  ~VectorPixelFiller() {}

  inline itk::VariableLengthVector<PixelValueType> operator()(const PixelValueType value, unsigned int vectorLength) const
  {
  itk::VariableLengthVector<PixelValueType> result;
  result.SetSize(vectorLength);
  result.Fill(value);
  return result;
  }
};

} // end namespace Functor
} // end namespace rtk

#endif // rtkFunctors_h
