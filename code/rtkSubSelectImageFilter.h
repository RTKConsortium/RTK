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
#ifndef __rtkSubSelectImageFilter_h
#define __rtkSubSelectImageFilter_h

#include <itkPasteImageFilter.h>
#include <itkExtractImageFilter.h>
#include "rtkConstantImageSource.h"
#include "rtkPhaseReader.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
template< typename ProjectionStackType>
class SubSelectImageFilter : public itk::ImageToImageFilter<ProjectionStackType, ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef SubSelectImageFilter             Self;
    typedef itk::ImageToImageFilter<ProjectionStackType, ProjectionStackType> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(SubSelectImageFilter, itk::ImageToImageFilter)

    /** The set of projections from which a subset will be extracted */
    void SetInputProjectionStack(const ProjectionStackType* Projections);
    typename ProjectionStackType::ConstPointer GetInputProjectionStack();

    typedef itk::PasteImageFilter<ProjectionStackType>                                  PasteFilterType;
    typedef itk::ExtractImageFilter<ProjectionStackType, ProjectionStackType>           ExtractFilterType;
    typedef rtk::ConstantImageSource<ProjectionStackType>                               EmptyProjectionStackSourceType;
    typedef rtk::ThreeDCircularProjectionGeometry                                       GeometryType;

    itkSetMacro(InputGeometry, GeometryType::Pointer)
    itkGetMacro(InputGeometry, GeometryType::Pointer)

    GeometryType::Pointer GetOutputGeometry();

protected:
    SubSelectImageFilter();
    ~SubSelectImageFilter(){}

    virtual void GenerateInputRequestedRegion();

    virtual void GenerateOutputInformation();

    /** Does the real work. */
    virtual void GenerateData();

    /** Member variables */
    GeometryType::Pointer     m_InputGeometry;
    GeometryType::Pointer     m_OutputGeometry;
    std::vector<bool>         m_SelectedProjections;
    int                       m_NbSelectedProjs;

private:
    SubSelectImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSubSelectImageFilter.txx"
#endif

#endif
