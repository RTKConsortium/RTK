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

#ifndef __rtkElektaSynergyRawImageFilter_h
#define __rtkElektaSynergyRawImageFilter_h

#include <itkUnaryFunctorImageFilter.h>
#include <itkConceptChecking.h>
#include <itkNumericTraits.h>

namespace rtk
{
namespace Function {

/** \class ElektaSynergyRawImageFilter
 * \brief Interprets the raw Elekta Synergy projection data to values.
 *
 * This filter does the same as rtk::ElektaSynergyLookupTableImageFilter:
 * converts the raw values measured by the panel to the logarithm of the value
 * divided by the maximum numerical value. This could be improved with a
 * calibration of the air value.
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template< class TInput, class TOutput>
class SynergyAttenuation
{
public:
  SynergyAttenuation()
    {
m_LogRef = log(TOutput(itk::NumericTraits<TInput>::max()-itk::NumericTraits<TInput>::min()+1));
    }
  ~SynergyAttenuation() {}
  bool operator!=( const SynergyAttenuation & ) const
    {
    return false;
    }
  bool operator==( const SynergyAttenuation & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A ) const
    {
    return log( TOutput(A+1) ) - m_LogRef;
    }

private:
  TOutput m_LogRef;
}; 
}

template <class TInputImage, class TOutputImage>
class ITK_EXPORT ElektaSynergyRawImageFilter :
    public
itk::UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                             Function::SynergyAttenuation<
  typename TInputImage::PixelType, 
  typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef ElektaSynergyRawImageFilter  Self;
  typedef itk::UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                       Function::SynergyAttenuation< typename TInputImage::PixelType,
                                                 typename TOutputImage::PixelType> >  Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyRawImageFilter,
               itk::UnaryFunctorImageFilter);

protected:
  ElektaSynergyRawImageFilter() {}
  virtual ~ElektaSynergyRawImageFilter() {}

private:
  ElektaSynergyRawImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace rtk


#endif
