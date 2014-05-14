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

#ifndef __rtkFourDConjugateGradientConeBeamReconstructionFilter_txx
#define __rtkFourDConjugateGradientConeBeamReconstructionFilter_txx

#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"

#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"

#include <algorithm>

namespace rtk
{

template<class VolumeSeriesType, class ProjectionStackType>
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::FourDConjugateGradientConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_NumberOfIterations=3;

  // Create the filters
  m_ZeroMultiplyFilter = MultiplyVolumeFilterType::New();
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_CGOperator = CGOperatorFilterType::New();
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());
  m_ProjStackToFourDFilter = ProjStackToFourDFilterType::New();

  // Set permanent parameters
  m_ZeroMultiplyFilter->SetConstant2(itk::NumericTraits<typename VolumeSeriesType::PixelType>::ZeroValue());

  // Set memory management parameters
  m_ZeroMultiplyFilter->ReleaseDataFlagOn();
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
    this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetInputProjectionStack(const VolumeType* Projection)
{
    this->SetNthInput(1, const_cast<VolumeType*>(Projection));
}

template<class VolumeSeriesType, class ProjectionStackType>
typename VolumeSeriesType::ConstPointer
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GetInputVolumeSeries()
{
    return static_cast< const VolumeSeriesType * >
            ( this->itk::ProcessObject::GetInput(0) );
}

template<class VolumeSeriesType, class ProjectionStackType>
typename ProjectionStackType::ConstPointer
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GetInputProjectionStack()
{
    return static_cast< const VolumeType * >
            ( this->itk::ProcessObject::GetInput(1) );
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetForwardProjectionFilter (int _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter( _arg );
    m_CGOperator->SetForwardProjectionFilter( m_ForwardProjectionFilter );
    }
}


template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetBackProjectionFilter (int _arg)
{
  if( _arg != this->GetBackProjectionFilter() )
    {
    Superclass::SetBackProjectionFilter( _arg );
    m_BackProjectionFilter = this->InstantiateBackProjectionFilter( _arg );
    m_CGOperator->SetBackProjectionFilter( m_BackProjectionFilter );

    m_BackProjectionFilterForB = this->InstantiateBackProjectionFilter( _arg );
    m_ProjStackToFourDFilter->SetBackProjectionFilter(m_BackProjectionFilterForB);
    }
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetWeights(const itk::Array2D<float> _arg)
{
  m_ProjStackToFourDFilter->SetWeights(_arg);
  m_CGOperator->SetWeights(_arg);
  this->Modified();
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_ZeroMultiplyFilter->SetInput1(this->GetInputVolumeSeries());
  m_CGOperator->SetInputProjectionStack(this->GetInputProjectionStack());
  m_ConjugateGradientFilter->SetX(this->GetInputVolumeSeries());

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_ProjStackToFourDFilter->SetInputVolumeSeries(m_ZeroMultiplyFilter->GetOutput());
  m_ProjStackToFourDFilter->SetInputProjectionStack(this->GetInputProjectionStack());
  m_ConjugateGradientFilter->SetB(m_ProjStackToFourDFilter->GetOutput());

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->m_Geometry);
  m_ProjStackToFourDFilter->SetGeometry(this->m_Geometry.GetPointer());

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);

  // Have the last filter calculate its output information
  std::cout << "In FourDConjugateGradientConeBeamReconstructionFilter. About to UpdateOutputInformation" << std::endl;
  m_ConjugateGradientFilter->UpdateOutputInformation();
  std::cout << "In FourDConjugateGradientConeBeamReconstructionFilter. UpdateOutputInformation complete" << std::endl;

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_ConjugateGradientFilter->GetOutput() );
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  std::cout << "In FourDConjugateGradientConeBeamReconstructionFilter : Entering GenerateData()" << std::endl;
  m_ConjugateGradientFilter->Update();
  this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
  std::cout << "In FourDConjugateGradientConeBeamReconstructionFilter : Leaving GenerateData()" << std::endl;
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::PrintTiming(std::ostream& os) const
{
}

} // end namespace rtk

#endif // __rtkFourDConjugateGradientConeBeamReconstructionFilter_txx
