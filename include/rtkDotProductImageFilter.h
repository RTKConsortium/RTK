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
#ifndef rtkDotProductImageFilter_h
#define rtkDotProductImageFilter_h

#include "rtkMacro.h"
#include "itkMetaProgrammingLibrary.h"
#include <itkEnableIf.h>
#if ITK_VERSION_MAJOR<5
# include <itkBinaryFunctorImageFilter.h>
#else
# include <itkBinaryGeneratorImageFilter.h>
#endif

namespace rtk
{
namespace Functor
{
  template< class TPixel, class TInternal>
  class DotProduct
  {
  public:
    DotProduct() {}
    ~DotProduct() {}
    bool operator!=(const DotProduct &) const
    {
      return false;
    }

    bool operator==(const DotProduct & other) const
    {
      return !( *this != other );
    }

  template<typename T = TPixel>
  inline
  typename itk::mpl::EnableIf<itk::mpl::IsSame<T, TInternal>, TInternal>::Type
  operator()(const TInternal & A, const TInternal & B) const
  { return static_cast<TInternal>( A * B ); }

  template<typename T = TPixel>
  typename itk::mpl::DisableIf<itk::mpl::IsSame<T, TInternal>, TInternal>::Type
  operator()(const TPixel & A, const TPixel & B) const
    {
    TInternal out = 0;
    for (unsigned int component=0; component < itk::NumericTraits<TPixel>::GetLength(A); component++)
      {
      out += A[component] * B[component];
      }
    return out;
    }
  };
} // end namespace functor

  /** \class DotProductImageFilter
   * \brief Computes the dot product between two itk::VectorImage
   *
   * This filter computes the dot product between two itk::VectorImage.
   * It will also accept itk::Image in input, in which case it will perform
   * like a itk::DotProductImageFilter, but slower.
   *
   * \author Cyril Mory
   */

template< typename TImage >
class ITK_EXPORT DotProductImageFilter:
  public
#if ITK_VERSION_MAJOR<5
    itk::BinaryFunctorImageFilter< TImage, TImage, TImage, typename Functor::DotProduct<TImage, double>  >
#else
    itk::BinaryGeneratorImageFilter< TImage, TImage, typename itk::Image<typename TImage::InternalPixelType, TImage::ImageDimension> >
#endif
{
public:
  /** Standard class typedefs. */
  typedef DotProductImageFilter           Self;
#if ITK_VERSION_MAJOR<5
  typedef itk::BinaryFunctorImageFilter< TImage, TImage, TImage, typename Functor::DotProduct<TImage, double>  > Superclass;
#else
  typedef itk::BinaryGeneratorImageFilter< TImage, TImage, typename itk::Image<typename TImage::InternalPixelType, TImage::ImageDimension> > Superclass;
#endif
  typedef itk::SmartPointer< Self >            Pointer;
  typedef itk::SmartPointer< const Self >      ConstPointer;

  using FunctorType =  Functor::DotProduct< typename TImage::PixelType,
                                            typename TImage::InternalPixelType>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(DotProductImageFilter,
#if ITK_VERSION_MAJOR<5
               itk::BinaryFunctorImageFilter);
#else
               itk::BinaryGeneratorImageFilter);
#endif

protected:
  DotProductImageFilter()
    {
#if !defined( ITK_WRAPPING_PARSER )
    Superclass::SetFunctor(FunctorType());
#endif
    }
  ~DotProductImageFilter() override {}

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(DotProductImageFilter);
};
} // end namespace rtk

#endif
