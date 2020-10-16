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

#ifndef rtkLUTbasedVariableI0RawToAttenuationImageFilter_h
#define rtkLUTbasedVariableI0RawToAttenuationImageFilter_h

#include <itkNumericTraits.h>
#include <itkSubtractImageFilter.h>
#include <itkLogImageFilter.h>
#include <itkThresholdImageFilter.h>

#include "rtkLookupTableImageFilter.h"

namespace rtk
{
/** \class LUTbasedVariableI0RawToAttenuationImageFilter
 * \brief Performs the conversion from raw data to attenuations
 *
 * Performs the conversion from raw data to attenuations using a lookup table
 * which is typically possible when the input type is 16-bit, e.g., unsigned
 * short. The I0 value (intensity when there is no attenuation) is assumed to
 * be constant and can be changed.
 *
 * If the input is of type I0EstimationProjectionFilter, then the member I0 is
 * not used but the estimated value is automatically retrieved.
 *
 * The lookup table is obtained using the following mini-pipeline:
 *
 * \dot
 * digraph LookupTable {
 *
 *  Input1 [label="index"]
 *  Input2 [label="log(max(1,m_I0-m_IDark))"]
 *  Input3 [label="m_IDark"]
 *  Output [label="LookupTable", shape=Mdiamond];
 *
 *  node [shape=box];
 *  SubR [label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"]
 *  ThresR [label="itk::ThresholdImageFilter" URL="\ref itk::ThresholdImageFilter"];
 *  Log [label="itk::LogImageFilter" URL="\ref itk::LogImageFilter"]
 *  Sub [label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 *
 *  Input1 -> SubR
 *  Input3 -> SubR
 *  SubR -> ThresR
 *  ThresR -> Log
 *
 *
 *  Log -> Sub
 *  Input2 -> Sub
 *  Sub -> Output
 * }
 * \enddot
 *
 * \test rtklutbasedrawtoattenuationtest.cxx
 *
 * \author S. Brousmiche, S. Rit
 *
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputImage, class TOutputImage>
class LUTbasedVariableI0RawToAttenuationImageFilter : public LookupTableImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(LUTbasedVariableI0RawToAttenuationImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(LUTbasedVariableI0RawToAttenuationImageFilter);
#endif

  /** Standard class type alias. */
  using Self = LUTbasedVariableI0RawToAttenuationImageFilter;
  using Superclass = LookupTableImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using InputImagePixelType = typename TInputImage::PixelType;
  using OutputImagePixelType = typename TOutputImage::PixelType;
  using LookupTableType = typename Superclass::FunctorType::LookupTableType;
  using SubtractLUTFilterType = typename itk::SubtractImageFilter<LookupTableType>;
  using ThresholdLUTFilterType = typename itk::ThresholdImageFilter<LookupTableType>;
  using LogLUTFilterType = typename itk::LogImageFilter<LookupTableType, LookupTableType>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LUTbasedVariableI0RawToAttenuationImageFilter, LookupTableImageFilter);

  /** Air level I0
   */
  itkGetMacro(I0, double);
  itkSetMacro(I0, double);

  /** Intensity when there is no photons (beam off)
   */
  itkGetMacro(IDark, double);
  itkSetMacro(IDark, double);

  void
  BeforeThreadedGenerateData() override;

protected:
  LUTbasedVariableI0RawToAttenuationImageFilter();
  ~LUTbasedVariableI0RawToAttenuationImageFilter() override = default;

private:
  double                                   m_I0;
  double                                   m_IDark;
  typename SubtractLUTFilterType::Pointer  m_SubtractRampFilter;
  typename ThresholdLUTFilterType::Pointer m_ThresholdRampFilter;
  typename LogLUTFilterType::Pointer       m_LogRampFilter;
  typename SubtractLUTFilterType::Pointer  m_SubtractLUTFilter;
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkLUTbasedVariableI0RawToAttenuationImageFilter.hxx"
#endif

#endif
