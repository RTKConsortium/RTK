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

#include "rtkMacro.h"

namespace rtk
{

namespace Function {  
  
/** \class ObiAttenuation
 * \brief Converts a raw value measured by the Varian OBI system to attenuation
 *
 * The user can specify I0 and IDark values. The defaults are 139000 and 0,
 * respectively.
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
    return (!A)?0.:TOutput( vcl_log(m_I0-m_IDark) - vcl_log( A-m_IDark ) );
    }
  void SetI0(double i0) {m_I0 = i0;}
  void SetIDark(double idark) {m_IDark = idark;}

private:
  double m_I0;
  double m_IDark;
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

  itkGetMacro(I0, double);
  itkSetMacro(I0, double);

  itkGetMacro(IDark, double);
  itkSetMacro(IDark, double);

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

protected:
  VarianObiRawImageFilter();
  ~VarianObiRawImageFilter() {}

private:
  VarianObiRawImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  double m_I0;
  double m_IDark;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkVarianObiRawImageFilter.hxx"
#endif

#endif
