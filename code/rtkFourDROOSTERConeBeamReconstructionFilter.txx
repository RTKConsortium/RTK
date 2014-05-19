#ifndef __rtkFourDROOSTERConeBeamReconstructionFilter_txx
#define __rtkFourDROOSTERConeBeamReconstructionFilter_txx

#include "rtkFourDROOSTERConeBeamReconstructionFilter.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType>
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::FourDROOSTERConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(3);

  // Set the default values of member parameters
  m_GammaSpace=1.0;
  m_GammaTime=1.0;
  m_TV_iterations=2;
  m_MainLoop_iterations=2;
  m_CG_iterations=2;

  // Create the filters
  m_FourDCGFilter = FourDCGFilterType::New();
  m_PositivityFilter = ThresholdFilterType::New();
  m_AverageOutOfROIFilter = AverageOutOfROIFilterType::New();
  m_TVDenoisingSpace = TVDenoisingFilterType::New();
  m_TVDenoisingTime = TVDenoisingFilterType::New();

  // Set permanent connections
  m_PositivityFilter->SetInput(m_FourDCGFilter->GetOutput());
  m_AverageOutOfROIFilter->SetInput(m_PositivityFilter->GetOutput());
//  m_AverageOutOfROIFilter->SetInput(m_FourDCGFilter->GetOutput());

  m_TVDenoisingSpace->SetInput(m_AverageOutOfROIFilter->GetOutput());
  m_TVDenoisingTime->SetInput(m_TVDenoisingSpace->GetOutput());

  // Set constant parameters
  m_DimensionsProcessedForTVSpace[0]=true;
  m_DimensionsProcessedForTVSpace[1]=true;
  m_DimensionsProcessedForTVSpace[2]=true;
  m_DimensionsProcessedForTVSpace[3]=false;
  m_TVDenoisingSpace->SetDimensionsProcessed(this->m_DimensionsProcessedForTVSpace);

  m_DimensionsProcessedForTVTime[0]=false;
  m_DimensionsProcessedForTVTime[1]=false;
  m_DimensionsProcessedForTVTime[2]=false;
  m_DimensionsProcessedForTVTime[3]=true;
  m_TVDenoisingTime->SetDimensionsProcessed(this->m_DimensionsProcessedForTVTime);
  m_TVDenoisingTime->SetBoundaryConditionToPeriodic();

  m_PositivityFilter->SetOutsideValue(0.0);
  m_PositivityFilter->ThresholdBelow(0.0);

  // Set memory management parameters
  m_FourDCGFilter->ReleaseDataFlagOn();
  m_PositivityFilter->ReleaseDataFlagOn();
  m_AverageOutOfROIFilter->ReleaseDataFlagOn();
  m_TVDenoisingSpace->ReleaseDataFlagOn();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetNthInput(1, const_cast<ProjectionStackType*>(Projection));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetInputROI(const VolumeType* ROI)
{
  this->SetNthInput(2, const_cast<VolumeType*>(ROI));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename VolumeSeriesType::ConstPointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename ProjectionStackType::Pointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetInputProjectionStack()
{
  return static_cast< ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::VolumeType::Pointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetInputROI()
{
  return static_cast< VolumeType * >
          ( this->itk::ProcessObject::GetInput(2) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetForwardProjectionFilter(int _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_FourDCGFilter->SetForwardProjectionFilter( _arg );
    }
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetBackProjectionFilter(int _arg)
{
  if( _arg != this->GetBackProjectionFilter() )
    {
    Superclass::SetBackProjectionFilter( _arg );
    m_FourDCGFilter->SetBackProjectionFilter( _arg );
    }
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetWeights(const itk::Array2D<float> _arg)
{
  m_FourDCGFilter->SetWeights(_arg);
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetROI(VolumeType* ROI)
{
  m_AverageOutOfROIFilter->SetSegmentation(ROI);
  this->Modified();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_FourDCGFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_FourDCGFilter->SetInputProjectionStack(this->GetInputProjectionStack());
  m_AverageOutOfROIFilter->SetROI(this->GetInputROI());

  // Set runtime parameters
  m_FourDCGFilter->SetGeometry(this->m_Geometry);
  m_FourDCGFilter->SetNumberOfIterations(this->m_CG_iterations);

  m_TVDenoisingSpace->SetNumberOfIterations(this->m_TV_iterations);
  m_TVDenoisingSpace->SetGamma(this->m_GammaSpace);

  m_TVDenoisingTime->SetNumberOfIterations(this->m_TV_iterations);
  m_TVDenoisingTime->SetGamma(this->m_GammaTime);

  // Have the last filter calculate its output information
  m_TVDenoisingTime->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_TVDenoisingTime->GetOutput() );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  for (int i=0; i<m_MainLoop_iterations; i++)
    {
    // After the first iteration, we need to use the output as input
    if (i>0)
      {
      typename VolumeSeriesType::Pointer pimg = m_TVDenoisingTime->GetOutput();
      pimg->DisconnectPipeline();
      m_FourDCGFilter->SetInputVolumeSeries(pimg);
      }

    m_CGProbe.Start();
    m_FourDCGFilter->Update();
    m_CGProbe.Stop();

    m_PositivityProbe.Start();
    m_PositivityFilter->Update();
    m_PositivityProbe.Stop();

    m_ROIProbe.Start();
    m_AverageOutOfROIFilter->Update();
    m_ROIProbe.Stop();

    m_TVSpaceProbe.Start();
    m_TVDenoisingSpace->Update();
    m_TVSpaceProbe.Stop();

    m_TVTimeProbe.Start();
    m_TVDenoisingTime->Update();
    m_TVTimeProbe.Stop();
    }

  this->GraftOutput( m_TVDenoisingTime->GetOutput() );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::PrintTiming(std::ostream& os) const
{
  os << "FourDROOSTERConeBeamReconstructionFilter timing:" << std::endl;
  os << "  4D conjugate gradient reconstruction: " << m_CGProbe.GetTotal()
     << ' ' << m_CGProbe.GetUnit() << std::endl;
  os << "  Positivity enforcement: " << m_PositivityProbe.GetTotal()
     << ' ' << m_PositivityProbe.GetUnit() << std::endl;
  os << "  Averaging along time outside the ROI where movement is allowed: " << m_ROIProbe.GetTotal()
     << ' ' << m_ROIProbe.GetUnit() << std::endl;
  os << "  Spatial total variation denoising: " << m_TVSpaceProbe.GetTotal()
     << ' ' << m_TVSpaceProbe.GetUnit() << std::endl;
  os << "  Temporal total variation denoising: " << m_TVTimeProbe.GetTotal()
     << ' ' << m_TVTimeProbe.GetUnit() << std::endl;
}

}// end namespace


#endif
