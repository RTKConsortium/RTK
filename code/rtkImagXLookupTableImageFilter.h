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

#ifndef __rtkImagXLookupTableImageFilter_h
#define __rtkImagXLookupTableImageFilter_h

#include "rtkLookupTableImageFilter.h"
#include <itkNumericTraits.h>

namespace rtk
{

/** \class ImagXLookupTableImageFilter
 * \brief Lookup table for ImagX data.
 *
 * The lookup table converts the raw values measured by the panel to the
 * logarithm of the value divided by the maximum numerical value. This could
 * be improved with a calibration of the air value.
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT ImagXLookupTableImageFilter : public LookupTableImageFilter<TInputImage, TOutputImage>
{

public:
  /** Standard class typedefs. */
  typedef ImagXLookupTableImageFilter                       Self;
  typedef LookupTableImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef typename TInputImage::PixelType                   InputImagePixelType;
  typedef typename TOutputImage::PixelType                  OutputImagePixelType;
  typedef typename Superclass::FunctorType::LookupTableType LookupTableType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ImagXLookupTableImageFilter, LookupTableImageFilter);
protected:
  ImagXLookupTableImageFilter();
  virtual ~ImagXLookupTableImageFilter() {
  }

private:
  ImagXLookupTableImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);              //purposely not implemented

};

} // end namespace rtk

template <class TInputImage, class TOutputImage>
rtk::ImagXLookupTableImageFilter<TInputImage, TOutputImage>::ImagXLookupTableImageFilter()
{
  // Create the lut
  typename LookupTableType::Pointer lut = LookupTableType::New();
  typename LookupTableType::SizeType size;
  size[0] = itk::NumericTraits<InputImagePixelType>::max()-itk::NumericTraits<InputImagePixelType>::min()+1;
  lut->SetRegions( size );
  lut->Allocate();

  OutputImagePixelType logRef = log(double(size[0]) );

  // Iterate and set lut
  itk::ImageRegionIteratorWithIndex<LookupTableType> it( lut, lut->GetBufferedRegion() );
  it.GoToBegin();
  while( !it.IsAtEnd() )
    {
      if( (logRef - log(it.GetIndex()[0]+1.)) <= 0. )
        it.Set (0.);
      else
        it.Set( logRef - log(it.GetIndex()[0]+1.) );
      ++it;
    }

  // Set the lut to member and functor
  this->SetLookupTable(lut);
}

#endif
