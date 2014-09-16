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
#ifndef __rtkPhaseGatingImageFilter_h
#define __rtkPhaseGatingImageFilter_h

#include <itkPasteImageFilter.h>
#include <itkExtractImageFilter.h>
#include "rtkConstantImageSource.h"
#include "rtkPhaseReader.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
template< typename ProjectionStackType>
class PhaseGatingImageFilter : public itk::ImageToImageFilter<ProjectionStackType, ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef PhaseGatingImageFilter             Self;
    typedef itk::ImageToImageFilter<ProjectionStackType, ProjectionStackType> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(PhaseGatingImageFilter, itk::ImageToImageFilter)

    /** The set of projections from which a subset will be extracted */
    void SetInputProjectionStack(const ProjectionStackType* Projections);

    typedef itk::PasteImageFilter<ProjectionStackType>                                  PasteFilterType;
    typedef itk::ExtractImageFilter<ProjectionStackType, ProjectionStackType>           ExtractFilterType;
    typedef rtk::ConstantImageSource<ProjectionStackType>                               EmptyProjectionStackSourceType;
    typedef rtk::ThreeDCircularProjectionGeometry                                       GeometryType;

    itkSetMacro(InputGeometry, GeometryType::Pointer)
    itkGetMacro(InputGeometry, GeometryType::Pointer)

    itkSetMacro(FileName, std::string)
    itkGetMacro(FileName, std::string)

    itkSetMacro(GatingWindowWidth, float)
    itkGetMacro(GatingWindowWidth, float)

    itkSetMacro(GatingWindowCenter, float)
    itkGetMacro(GatingWindowCenter, float)

    itkSetMacro(GatingWindowShape, int)
    itkGetMacro(GatingWindowShape, int)

    std::vector<float> GetGatingWeights();
    std::vector<float> GetGatingWeightsOnSelectedProjections();

    GeometryType::Pointer GetOutputGeometry();

protected:
    PhaseGatingImageFilter();
    ~PhaseGatingImageFilter(){}

    typename ProjectionStackType::ConstPointer GetInputProjectionStack();

    virtual void GenerateInputRequestedRegion();

    virtual void GenerateOutputInformation();

    void SelectProjections();

    void ComputeWeights();

    void SetPhases(std::vector<float> phases);

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    rtk::PhaseReader::Pointer m_PhaseReader;

    /** Member variables */
    GeometryType::Pointer     m_InputGeometry;
    GeometryType::Pointer     m_OutputGeometry;
    std::vector<float>        m_GatingWeights;
    std::vector<float>        m_GatingWeightsOnSelectedProjections;
    std::vector<float>        m_Phases;
    std::vector<bool>         m_SelectedProjections;
    int                       m_NbSelectedProjs;
    float                     m_GatingWindowWidth;
    float                     m_GatingWindowCenter;
    int                       m_GatingWindowShape;
    std::string               m_FileName;

private:
    PhaseGatingImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkPhaseGatingImageFilter.txx"
#endif

#endif
