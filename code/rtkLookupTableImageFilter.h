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

#ifndef __rtkLookupTableImageFilter_h
#define __rtkLookupTableImageFilter_h

#include <itkUnaryFunctorImageFilter.h>

namespace rtk
{

namespace Functor
{
/** \brief Function to do the lookup operation.
 *
 * The lookup table is a 1D image which must contain all possible values.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template< class TInput, class TOutput >
class LUT
{
public:
  typedef itk::Image<TOutput,1>                LookupTableType;
  typedef typename LookupTableType::PixelType* LookupTableDataPointerType;

  LUT() {};
  ~LUT() {};

  /** Get/Set the lookup table. */
  LookupTableDataPointerType GetLookupTableDataPointer() {
    return m_LookupTableDataPointer;
  }
  void SetLookupTableDataPointer(LookupTableDataPointerType lut) {
    m_LookupTableDataPointer = lut;
  }

  bool operator!=( const LUT & lut ) const {
    return m_LookupTableDataPointer != lut->GetLookupTableDataPointer;
  }
  bool operator==( const LUT & lut ) const {
    return m_LookupTableDataPointer == lut->GetLookupTableDataPointer;
  }

  inline TOutput operator()( const TInput & val ) const {
    return m_LookupTableDataPointer[val];
  }

private:
  LookupTableDataPointerType m_LookupTableDataPointer;
};
} // end namespace Functor

/** \class LookupTableImageFilter
 * \brief Converts integer values of an input image using lookup table.
 *
 * The lookup table is passed via a functor of type Functor::LUT
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT LookupTableImageFilter : public
  itk::UnaryFunctorImageFilter< TInputImage,
                                TOutputImage,
                                Functor::LUT< typename TInputImage::PixelType,
                                              typename TOutputImage::PixelType> >
{

public:
  /** Lookup table type definition. */
  typedef Functor::LUT< typename TInputImage::PixelType, typename TOutputImage::PixelType > FunctorType;
  typedef typename FunctorType::LookupTableType                                                     LookupTableType;

  /** Standard class typedefs. */
  typedef LookupTableImageFilter                                                        Self;
  typedef itk::UnaryFunctorImageFilter<TInputImage, TOutputImage, FunctorType > Superclass;
  typedef itk::SmartPointer<Self>                                               Pointer;
  typedef itk::SmartPointer<const Self>                                         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(LookupTableImageFilter, itk::UnaryFunctorImageFilter);

  /** Set lookup table. */
  virtual void SetLookupTable(LookupTableType* _arg) {
    //Idem as itkSetObjectMacro + call to functor SetLookupTableDataPointer
    itkDebugMacro("setting " << "LookupTable" " to " << _arg );
    if (this->m_LookupTable != _arg) {
      this->m_LookupTable = _arg;
      this->Modified();
      this->GetFunctor().SetLookupTableDataPointer(_arg->GetBufferPointer() );
      }
  }

  /** Get lookup table. */
  itkGetObjectMacro(LookupTable, LookupTableType);

protected:
  LookupTableImageFilter() {}
  virtual ~LookupTableImageFilter() {}
  typename LookupTableType::Pointer m_LookupTable;

private:
  LookupTableImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace rtk

#endif
