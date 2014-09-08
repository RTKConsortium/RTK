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

#ifndef __rtkSoftThresholdImageFilter_h
#define __rtkSoftThresholdImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "itkConceptChecking.h"
#include "itkSimpleDataObjectDecorator.h"

namespace rtk
{
  
/** \class SoftThresholdImageFilter
 *
 * \brief Soft thresholds an image
 *
 * This filter produces an output image whose pixels
 * are max(x-t,0).sign(x) where x is the corresponding
 * input pixel value and t the threshold
 *
 * \ingroup IntensityImageFilters  Multithreaded
 */
using namespace itk;

namespace Functor {  
  
template< class TInput, class TOutput>
class SoftThreshold
{
public:
  SoftThreshold()
    {
    m_Threshold = NumericTraits<TInput>::Zero;
    }
  ~SoftThreshold() {};

  void SetThreshold( const TInput & thresh )
    { m_Threshold = thresh; }

  bool operator!=( const SoftThreshold & other ) const
    {
    if( m_Threshold != other.m_Threshold )
      {
      return true;
      }
    return false;
    }
  bool operator==( const SoftThreshold & other ) const
    {
    return !(*this != other);
    }

  inline TOutput operator()( const TInput & A ) const
    {
    return (vnl_math_sgn(A) * vnl_math_max((TInput)vnl_math_abs(A) - m_Threshold, (TInput)0.0));
    }

private:
  TInput      m_Threshold;

};
}

template <class TInputImage, class TOutputImage>
class ITK_EXPORT SoftThresholdImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                        Functor::SoftThreshold<
  typename TInputImage::PixelType, 
  typename TOutputImage::PixelType> >
{
public:
  /** Standard class typedefs. */
  typedef SoftThresholdImageFilter                            Self;
  typedef UnaryFunctorImageFilter
  <TInputImage,TOutputImage,
  Functor::SoftThreshold< typename TInputImage::PixelType,
                          typename TOutputImage::PixelType> > Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SoftThresholdImageFilter, UnaryFunctorImageFilter);

  /** Pixel types. */
  typedef typename TInputImage::PixelType  InputPixelType;
  typedef typename TOutputImage::PixelType OutputPixelType;

  /** Type of DataObjects to use for scalar inputs */
  typedef SimpleDataObjectDecorator<InputPixelType> InputPixelObjectType;

  /** Set the threshold */
  virtual void SetThreshold(const InputPixelType threshold);
                 
#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(OutputEqualityComparableCheck,
                  (Concept::EqualityComparable<OutputPixelType>));
  itkConceptMacro(InputPixelTypeComparable,
                  (Concept::Comparable<InputPixelType>));
  itkConceptMacro(InputOStreamWritableCheck,
                  (Concept::OStreamWritable<InputPixelType>));
  itkConceptMacro(OutputOStreamWritableCheck,
                  (Concept::OStreamWritable<OutputPixelType>));
  /** End concept checking */
#endif

protected:
  SoftThresholdImageFilter();
  virtual ~SoftThresholdImageFilter() {}

private:
  SoftThresholdImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSoftThresholdImageFilter.txx"
#endif

#endif
