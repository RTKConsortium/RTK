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

#ifndef rtkElektaSynergyLookupTableImageFilter_h
#define rtkElektaSynergyLookupTableImageFilter_h

#include "rtkLookupTableImageFilter.h"
#include <itkNumericTraits.h>

namespace rtk
{

/** \class ElektaSynergyLookupTableImageFilter
 * \brief Lookup table for Elekta Synergy data.
 *
 * The lookup table converts the raw values measured by the panel after a first
 * raw lookup table to the logarithm of the value divided by the maximum
 * numerical value. This could be improved with a calibration of the air value.
 *
 * \test rtkelektatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template <class TOutputImage>
class ITK_EXPORT ElektaSynergyLookupTableImageFilter:
    public LookupTableImageFilter< itk::Image<unsigned short, TOutputImage::ImageDimension>,
                                   TOutputImage >
{

public:
  /** Standard class typedefs. */
  typedef ElektaSynergyLookupTableImageFilter                                 Self;
  typedef LookupTableImageFilter<itk::Image<unsigned short, 
                                            TOutputImage::ImageDimension>,
                                 TOutputImage>                                Superclass;
  typedef itk::SmartPointer<Self>                                             Pointer;
  typedef itk::SmartPointer<const Self>                                       ConstPointer;

  typedef unsigned short                                    InputImagePixelType;
  typedef typename TOutputImage::PixelType                  OutputImagePixelType;
  typedef typename Superclass::FunctorType::LookupTableType LookupTableType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyLookupTableImageFilter, LookupTableImageFilter);

protected:
  ElektaSynergyLookupTableImageFilter();
  ~ElektaSynergyLookupTableImageFilter() {}

private:
  ElektaSynergyLookupTableImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);              //purposely not implemented

};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkElektaSynergyLookupTableImageFilter.hxx"
#endif

#endif
