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

#ifndef rtkVarianObiRawImageFilter_h
#define rtkVarianObiRawImageFilter_h

#include <itkUnaryFunctorImageFilter.h>
#include <itkConceptChecking.h>
#include <itkNumericTraits.h>

#define HND_INTENSITY_MAX (139000.)

namespace rtk
{
  
namespace Function {  
  
/** \class ObiAttenuation
 * \brief Converts a raw value measured by the Varian OBI system to attenuation
 *
 * The current implementation assues a maximum possible value for hnd of 139000.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template< class TInput, class TOutput>
class ObiAttenuation
{
public:
  ObiAttenuation() {}
  ~ObiAttenuation() {}
  bool operator!=( const ObiAttenuation & ) const
    {
    return false;
    }
  bool operator==( const ObiAttenuation & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A ) const
    {
    return (!A)?0.:TOutput( vcl_log(HND_INTENSITY_MAX ) - vcl_log( double(A) ) );
    }
}; 
}

/** \class VarianObiRawImageFilter
 * \brief Converts raw images measured by the Varian OBI system to attenuation
 *
 * Uses ObiAttenuation.
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT VarianObiRawImageFilter :
    public
itk::UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                             Function::ObiAttenuation<
  typename TInputImage::PixelType, 
  typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef VarianObiRawImageFilter  Self;
  typedef itk::UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                       Function::ObiAttenuation< typename TInputImage::PixelType,
                                                      typename TOutputImage::PixelType> >  Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(VarianObiRawImageFilter,
               itk::UnaryFunctorImageFilter);

protected:
  VarianObiRawImageFilter() {}
  ~VarianObiRawImageFilter() ITK_OVERRIDE {}

private:
  VarianObiRawImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace rtk


#endif
