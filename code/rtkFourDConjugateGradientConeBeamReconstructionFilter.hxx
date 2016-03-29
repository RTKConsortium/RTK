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

#ifndef __rtkFourDConjugateGradientConeBeamReconstructionFilter_hxx
#define __rtkFourDConjugateGradientConeBeamReconstructionFilter_hxx

#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"

#include <algorithm>

#include <itkImageFileWriter.h>

namespace rtk
{

template<class VolumeSeriesType, class ProjectionStackType>
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::FourDConjugateGradientConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_NumberOfIterations=3;
  m_CudaConjugateGradient = false; // 4D volumes of usual size only fit on the largest GPUs

  // Create the filters
  m_CGOperator = CGOperatorFilterType::New();
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_ProjStackToFourDFilter = ProjStackToFourDFilterType::New();

  // Memory management options
  GetProjectionStackToFourDFilter()->ReleaseDataFlagOn();
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
    GetConjugateGradientOperator()->SetForwardProjectionFilter( m_ForwardProjectionFilter );
    }
  if (_arg == 2) // The forward projection filter runs on GPU. It is most efficient to also run the interpolation on GPU, and to use GPU constant image sources
    {
    GetConjugateGradientOperator()->SetUseCudaInterpolation(true);
    GetConjugateGradientOperator()->SetUseCudaSources(true);
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
    GetConjugateGradientOperator()->SetBackProjectionFilter( m_BackProjectionFilter );

    m_BackProjectionFilterForB = this->InstantiateBackProjectionFilter( _arg );
    GetProjectionStackToFourDFilter()->SetBackProjectionFilter(m_BackProjectionFilterForB);
    }
  if (_arg == 2) // The back projection filter runs on GPU. It is most efficient to also run the splat on GPU, and to use GPU constant image sources
    {
    GetConjugateGradientOperator()->SetUseCudaSplat(true);
    GetConjugateGradientOperator()->SetUseCudaSources(true);
    GetProjectionStackToFourDFilter()->SetUseCudaSplat(true);
    GetProjectionStackToFourDFilter()->SetUseCudaSources(true);
    }
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetWeights(const itk::Array2D<float> _arg)
{
  GetProjectionStackToFourDFilter()->SetWeights(_arg);
  GetConjugateGradientOperator()->SetWeights(_arg);
  this->Modified();
}

template<class VolumeSeriesType, class ProjectionStackType>
typename rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::ProjStackToFourDFilterType*
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetProjectionStackToFourDFilter()
{
return(this->m_ProjStackToFourDFilter.GetPointer());
}

template<class VolumeSeriesType, class ProjectionStackType>
typename rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::CGOperatorFilterType*
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetConjugateGradientOperator()
{
return(this->m_CGOperator.GetPointer());
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  // Set the Conjugate Gradient filter (either on CPU or GPU depending on user's choice)
#ifdef RTK_USE_CUDA
  if (m_CudaConjugateGradient)
    m_ConjugateGradientFilter = rtk::CudaConjugateGradientImageFilter_4f::New();
#endif
  m_ConjugateGradientFilter->SetA(GetConjugateGradientOperator());

  // Set runtime connections
  GetConjugateGradientOperator()->SetInputProjectionStack(this->GetInputProjectionStack());
  m_ConjugateGradientFilter->SetX(this->GetInputVolumeSeries());

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  GetProjectionStackToFourDFilter()->SetInputVolumeSeries(this->GetInputVolumeSeries());
  GetProjectionStackToFourDFilter()->SetInputProjectionStack(this->GetInputProjectionStack());
  m_ConjugateGradientFilter->SetB(GetProjectionStackToFourDFilter()->GetOutput());

  // For the same reason, set geometry now
  GetConjugateGradientOperator()->SetGeometry(this->m_Geometry);
  GetProjectionStackToFourDFilter()->SetGeometry(this->m_Geometry.GetPointer());

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);

  // Have the last filter calculate its output information
  m_ConjugateGradientFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_ConjugateGradientFilter->GetOutput() );
}


template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateInputRequestedRegion()
{
  this->GetProjectionStackToFourDFilter()->PropagateRequestedRegion(this->GetProjectionStackToFourDFilter()->GetOutput());
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  GetProjectionStackToFourDFilter()->Update();

  // If GetProjectionStackToFourDFilter()->GetOutput() is stored in an itk::CudaImage, make sure its data is transferred on the CPU
  this->GetProjectionStackToFourDFilter()->GetOutput()->GetBufferPointer();

  m_ConjugateGradientFilter->Update();

  // Simply grafting the output of m_ConjugateGradientFilter to the main output
  // is sufficient in most cases, but when this output is then disconnected and replugged,
  // several images end up having the same CudaDataManager. The following solution is a
  // workaround for this problem
  typename VolumeSeriesType::Pointer pimg = m_ConjugateGradientFilter->GetOutput();
  pimg->DisconnectPipeline();

  this->GraftOutput( pimg);
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::PrintTiming(std::ostream& os) const
{
}

} // end namespace rtk

#endif // __rtkFourDConjugateGradientConeBeamReconstructionFilter_hxx
