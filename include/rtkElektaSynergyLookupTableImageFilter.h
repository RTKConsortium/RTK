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
 * \ingroup RTK ImageToImageFilter
 */
template <class TOutputImage>
class ITK_EXPORT ElektaSynergyLookupTableImageFilter
  : public LookupTableImageFilter<itk::Image<unsigned short, TOutputImage::ImageDimension>, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ElektaSynergyLookupTableImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ElektaSynergyLookupTableImageFilter);
#endif

  /** Standard class type alias. */
  using Self = ElektaSynergyLookupTableImageFilter;
  using Superclass = LookupTableImageFilter<itk::Image<unsigned short, TOutputImage::ImageDimension>, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using InputImagePixelType = unsigned short;
  using OutputImagePixelType = typename TOutputImage::PixelType;
  using LookupTableType = typename Superclass::FunctorType::LookupTableType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyLookupTableImageFilter, LookupTableImageFilter);

protected:
  ElektaSynergyLookupTableImageFilter();
  ~ElektaSynergyLookupTableImageFilter() override = default;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkElektaSynergyLookupTableImageFilter.hxx"
#endif

#endif
