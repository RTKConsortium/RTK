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
#ifndef rtkPhaseGatingImageFilter_h
#define rtkPhaseGatingImageFilter_h

#include "rtkSubSelectImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkPhaseReader.h"

namespace rtk
{
template< typename ProjectionStackType>
class PhaseGatingImageFilter : public SubSelectImageFilter<ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef PhaseGatingImageFilter                    Self;
    typedef SubSelectImageFilter<ProjectionStackType> Superclass;
    typedef itk::SmartPointer< Self >                 Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(PhaseGatingImageFilter, SubSelectImageFilter)

    itkSetMacro(PhasesFileName, std::string)
    itkGetMacro(PhasesFileName, std::string)

    itkSetMacro(GatingWindowWidth, float)
    itkGetMacro(GatingWindowWidth, float)

    itkSetMacro(GatingWindowCenter, float)
    itkGetMacro(GatingWindowCenter, float)

    itkSetMacro(GatingWindowShape, int)
    itkGetMacro(GatingWindowShape, int)

    std::vector<float> GetGatingWeights();
    std::vector<float> GetGatingWeightsOnSelectedProjections();

protected:
    PhaseGatingImageFilter();
    ~PhaseGatingImageFilter() {}

    void GenerateOutputInformation() ITK_OVERRIDE;

    void SelectProjections();

    void ComputeWeights();

    void SetPhases(std::vector<float> phases);

    /** Member pointers to the filters used internally (for convenience)*/
    rtk::PhaseReader::Pointer m_PhaseReader;

    /** Member variables */
    std::vector<float>        m_GatingWeights;
    std::vector<float>        m_GatingWeightsOnSelectedProjections;
    std::vector<float>        m_Phases;
    float                     m_GatingWindowWidth;
    float                     m_GatingWindowCenter;
    int                       m_GatingWindowShape;
    std::string               m_PhasesFileName;

private:
    PhaseGatingImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkPhaseGatingImageFilter.hxx"
#endif

#endif
