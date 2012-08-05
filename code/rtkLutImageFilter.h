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

#ifndef __rtkLutImageFilter_h
#define __rtkLutImageFilter_h

#include <itkUnaryFunctorImageFilter.h>

/** \class LutImageFilter
 * \brief TODO
 *
 * TODO
 *
 * \author Simon Rit
 *
 * \ingroup UnaryFunctorImageFilter
 */

namespace rtk
{

namespace Functor
{
template< class TInput, class TOutput >
class LUT
{
public:
  typedef itk::Image<TOutput,1>        LutType;
  typedef typename LutType::PixelType* LutDataPointerType;

  LUT() {};
  ~LUT() {};

  LutDataPointerType GetLutDataPointer() {
    return m_LutDataPointer;
  }
  void SetLutDataPointer(LutDataPointerType lut) {
    m_LutDataPointer = lut;
  }

  bool operator!=( const LUT & lut ) const {
    return m_LutDataPointer != lut->GetLutDataPointer;
  }
  bool operator==( const LUT & lut ) const {
    return m_LutDataPointer == lut->GetLutDataPointer;
  }

  inline TOutput operator()( const TInput & val ) const {
    return m_LutDataPointer[val];
  }

private:
  LutDataPointerType m_LutDataPointer;
};
} // end namespace Functor

template <class TInputImage, class TOutputImage>
class ITK_EXPORT LutImageFilter : public
  itk::UnaryFunctorImageFilter< TInputImage,
                                TOutputImage,
                                Functor::LUT< typename TInputImage::PixelType,
                                              typename TOutputImage::PixelType> >
{

public:
  /** Lookup table type definition. */
  typedef Functor::LUT< typename TInputImage::PixelType, typename TOutputImage::PixelType > FunctorType;
  typedef typename FunctorType::LutType                                                     LutType;

  /** Standard class typedefs. */
  typedef LutImageFilter                                                        Self;
  typedef itk::UnaryFunctorImageFilter<TInputImage, TOutputImage, FunctorType > Superclass;
  typedef itk::SmartPointer<Self>                                               Pointer;
  typedef itk::SmartPointer<const Self>                                         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(LutImageFilter, itk::UnaryFunctorImageFilter);

  /** Set lookup table. */
  virtual void SetLut(LutType* _arg) {
    //Idem as itkSetObjectMacro + call to functor SetLutDataPointer
    itkDebugMacro("setting " << "Lut" " to " << _arg );
    if (this->m_Lut != _arg) {
      this->m_Lut = _arg;
      this->Modified();
      this->GetFunctor().SetLutDataPointer(_arg->GetBufferPointer() );
      }
  }

  /** Get lookup table. */
  itkGetObjectMacro(Lut, LutType);

protected:
  LutImageFilter() {}
  virtual ~LutImageFilter() {}
  typename LutType::Pointer m_Lut;

private:
  LutImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace rtk

#endif
