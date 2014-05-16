#ifndef __rtkFourDReconstructionConjugateGradientOperator_txx
#define __rtkFourDReconstructionConjugateGradientOperator_txx

#include "rtkFourDReconstructionConjugateGradientOperator.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType>
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::FourDReconstructionConjugateGradientOperator()
{
    this->SetNumberOfRequiredInputs(2);

    // Create the two filters
    m_FourDToProjectionStackFilter = FourDToProjectionStackFilterType::New();
    m_ProjectionStackToFourDFilter = ProjectionStackToFourDFilterType::New();

    // Connect them
    m_ProjectionStackToFourDFilter->SetInputProjectionStack(m_FourDToProjectionStackFilter->GetOutput());
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
    this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::SetInputProjectionStack(const ProjectionStackType* Projection)
{
    this->SetNthInput(1, const_cast<ProjectionStackType*>(Projection));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename VolumeSeriesType::ConstPointer FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::GetInputVolumeSeries()
{
    return static_cast< const VolumeSeriesType * >
            ( this->itk::ProcessObject::GetInput(0) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename ProjectionStackType::ConstPointer FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::GetInputProjectionStack()
{
    return static_cast< const ProjectionStackType * >
            ( this->itk::ProcessObject::GetInput(1) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg)
{
    m_ProjectionStackToFourDFilter->SetBackProjectionFilter(_arg);
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg)
{
    m_FourDToProjectionStackFilter->SetForwardProjectionFilter(_arg);
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::SetWeights(const itk::Array2D<float> _arg)
{
    m_ProjectionStackToFourDFilter->SetWeights(_arg);
    m_FourDToProjectionStackFilter->SetWeights(_arg);
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg)
{
    m_ProjectionStackToFourDFilter->SetGeometry(_arg);
    m_FourDToProjectionStackFilter->SetGeometry(_arg);
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_FourDToProjectionStackFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_FourDToProjectionStackFilter->SetInputProjectionStack(this->GetInputProjectionStack());
  m_ProjectionStackToFourDFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());

  // Have the last filter calculate its output information
  m_ProjectionStackToFourDFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_ProjectionStackToFourDFilter->GetOutput() );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  m_ProjectionStackToFourDFilter->Update();
  this->GraftOutput( m_ProjectionStackToFourDFilter->GetOutput() );
}

}// end namespace


#endif
