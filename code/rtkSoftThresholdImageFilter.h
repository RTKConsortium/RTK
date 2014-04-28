/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    softThresholdImageFilter.h
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __softThresholdImageFilter_h
#define __softThresholdImageFilter_h

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
  typedef SoftThresholdImageFilter  Self;
  typedef UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                  Functor::SoftThreshold<
    typename TInputImage::PixelType, 
    typename TOutputImage::PixelType>   
  >                                   Superclass;
  typedef SmartPointer<Self>          Pointer;
  typedef SmartPointer<const Self>    ConstPointer;

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
