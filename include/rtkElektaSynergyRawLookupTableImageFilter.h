/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkElektaSynergyRawLookupTableImageFilter_h
#define rtkElektaSynergyRawLookupTableImageFilter_h

#include "rtkLookupTableImageFilter.h"
#include <itkNumericTraits.h>

namespace rtk
{

/** \class ElektaSynergyRawLookupTableImageFilter
 * \brief First part of ElektaSynergyLookupTableImageFilter.
 *
 * The lookup table has been split in two to allow application of the scatter
 * correction algorithm in between. The first part only set values to have max
 * value in air.
 *
 * \test rtkelektatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */
template <class TInputImage = itk::Image<unsigned short, 2>, class TOutputImage = itk::Image<unsigned short, 2>>
class ITK_TEMPLATE_EXPORT ElektaSynergyRawLookupTableImageFilter
  : public LookupTableImageFilter<TInputImage, TOutputImage>
{

public:
  ITK_DISALLOW_COPY_AND_MOVE(ElektaSynergyRawLookupTableImageFilter);

  /** Standard class type alias. */
  using Self = ElektaSynergyRawLookupTableImageFilter;
  using Superclass = LookupTableImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using InputImagePixelType = typename TInputImage::PixelType;
  using OutputImagePixelType = typename TOutputImage::PixelType;
  using LookupTableType = typename Superclass::FunctorType::LookupTableType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(ElektaSynergyRawLookupTableImageFilter);

  // Begin concept checking
  itkConceptMacro(SameTypeCheck, (itk::Concept::SameType<InputImagePixelType, unsigned short>));

protected:
  ElektaSynergyRawLookupTableImageFilter();
  ~ElektaSynergyRawLookupTableImageFilter() override = default;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkElektaSynergyRawLookupTableImageFilter.hxx"
#endif

#endif
