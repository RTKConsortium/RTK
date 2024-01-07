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
#ifndef rtkPhaseGatingImageFilter_h
#define rtkPhaseGatingImageFilter_h

#include "rtkSubSelectImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkPhaseReader.h"

namespace rtk
{
/** \class PhaseGatingImageFilter
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <typename ProjectionStackType>
class ITK_TEMPLATE_EXPORT PhaseGatingImageFilter : public SubSelectImageFilter<ProjectionStackType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(PhaseGatingImageFilter);

  /** Standard class type alias. */
  using Self = PhaseGatingImageFilter;
  using Superclass = SubSelectImageFilter<ProjectionStackType>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(PhaseGatingImageFilter);
#else
  itkTypeMacro(PhaseGatingImageFilter, SubSelectImageFilter);
#endif

  itkSetMacro(PhasesFileName, std::string);
  itkGetMacro(PhasesFileName, std::string);

  itkSetMacro(GatingWindowWidth, float);
  itkGetMacro(GatingWindowWidth, float);

  itkSetMacro(GatingWindowCenter, float);
  itkGetMacro(GatingWindowCenter, float);

  itkSetMacro(GatingWindowShape, int);
  itkGetMacro(GatingWindowShape, int);

  std::vector<float>
  GetGatingWeights();
  std::vector<float>
  GetGatingWeightsOnSelectedProjections();

protected:
  PhaseGatingImageFilter();
  ~PhaseGatingImageFilter() override = default;

  void
  GenerateOutputInformation() override;

  void
  SelectProjections();

  void
  ComputeWeights();

  void
  SetPhases(std::vector<float> phases);

  /** Member pointers to the filters used internally (for convenience)*/
  rtk::PhaseReader::Pointer m_PhaseReader;

  /** Member variables */
  std::vector<float> m_GatingWeights;
  std::vector<float> m_GatingWeightsOnSelectedProjections;
  std::vector<float> m_Phases;
  float              m_GatingWindowWidth;
  float              m_GatingWindowCenter;
  int                m_GatingWindowShape;
  std::string        m_PhasesFileName;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkPhaseGatingImageFilter.hxx"
#endif

#endif
