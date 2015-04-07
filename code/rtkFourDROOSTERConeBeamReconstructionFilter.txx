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
  m_PerformWarping=false;
  m_PhaseShift = 0;

  // Create the filters
  m_FourDCGFilter = FourDCGFilterType::New();
  m_PositivityFilter = ThresholdFilterType::New();
  m_AverageOutOfROIFilter = AverageOutOfROIFilterType::New();
  m_TVDenoisingTime = TemporalTVDenoisingFilterType::New();
#ifdef RTK_USE_CUDA
  m_AverageOutOfROIFilter = rtk::CudaAverageOutOfROIImageFilter::New();
  m_TVDenoisingTime = rtk::CudaLastDimensionTVDenoisingImageFilter::New();
#endif
  m_TVDenoisingSpace = SpatialTVDenoisingFilterType::New();
  m_Warp = WarpSequenceFilterType::New();
  m_Unwarp = UnwarpSequenceFilterType::New();

  // Set permanent connections
  m_PositivityFilter->SetInput(m_FourDCGFilter->GetOutput());
  m_AverageOutOfROIFilter->SetInput(m_PositivityFilter->GetOutput());
  m_TVDenoisingSpace->SetInput(m_AverageOutOfROIFilter->GetOutput());

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
  m_PositivityFilter->SetInPlace(true);
  m_AverageOutOfROIFilter->ReleaseDataFlagOn();
  m_TVDenoisingSpace->ReleaseDataFlagOn();
  m_Warp->ReleaseDataFlagOn();
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
::SetMotionMask(const VolumeType* mask)
{
  this->SetNthInput(2, const_cast<VolumeType*>(mask));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetDisplacementField(const MVFSequenceImageType* MVFs)
{
  this->SetNthInput(3, const_cast<MVFSequenceImageType*>(MVFs));
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
typename FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::MVFSequenceImageType::Pointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetDisplacementField()
{
  return static_cast< MVFSequenceImageType * >
          ( this->itk::ProcessObject::GetInput(3) );
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
::PreparePipeline()
{
  m_AverageOutOfROIFilter->SetROI(this->GetInputROI());

  m_FourDCGFilter->SetNumberOfIterations(this->m_CG_iterations);

  m_TVDenoisingSpace->SetNumberOfIterations(this->m_TV_iterations);
  m_TVDenoisingSpace->SetGamma(this->m_GammaSpace);

  m_TVDenoisingTime->SetNumberOfIterations(this->m_TV_iterations);
  m_TVDenoisingTime->SetGamma(this->m_GammaTime);

  // If requested, plug the warp filters into the pipeline
  if (m_PerformWarping)
    {
    m_Warp->SetDisplacementField(this->GetDisplacementField());
    m_Warp->SetPhaseShift(m_PhaseShift);

    m_TVDenoisingTime->SetInput(m_Warp->GetOutput());
    m_TVDenoisingTime->ReleaseDataFlagOn();

    m_Unwarp->SetInput(0, m_TVDenoisingTime->GetOutput());
    m_Unwarp->SetDisplacementField(this->GetDisplacementField());
    m_Unwarp->SetPhaseShift(m_PhaseShift);
    m_Unwarp->SetNumberOfIterations(4);
    }
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  //Get pointers to the input and output
  typename VolumeSeriesType::Pointer input0Ptr  = const_cast<VolumeSeriesType *>(this->GetInput(0));
  typename ProjectionStackType::Pointer input1Ptr  = this->GetInputProjectionStack();
  typename VolumeType::Pointer input2Ptr  = this->GetInputROI();

  input0Ptr->SetRequestedRegionToLargestPossibleRegion();
  input1Ptr->SetRequestedRegionToLargestPossibleRegion();
  input2Ptr->SetRequestedRegionToLargestPossibleRegion();

  if (m_PerformWarping)
    {
    typename MVFSequenceImageType::Pointer input3Ptr  = this->GetDisplacementField();

    input3Ptr->SetRequestedRegionToLargestPossibleRegion();
    }
  else
    {
    // If the filter is used with m_PerformWarping = true, then with
    // m_PerformWarping = false, it keeps requesting a region of the
    // input DVF, which by default may be larger than the largest
    // possible region of the DVF (it is the largest possible region of
    // the first input, and the sizes do not necessarily match).
    // This occurs, for example, in the fourdroostercudatest.
    this->RemoveInput(3);
    }
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  this->PreparePipeline();

  // Set some runtime connections
  m_FourDCGFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_FourDCGFilter->SetInputProjectionStack(this->GetInputProjectionStack());

  // Set runtime parameters
  m_FourDCGFilter->SetGeometry(this->m_Geometry);

  // If requested, plug the warp filters into the pipeline
  if (m_PerformWarping)
    {
      m_Warp->SetInput(0, m_TVDenoisingSpace->GetOutput());

      // Have the last filter calculate its output information
      m_Unwarp->UpdateOutputInformation();

      // Copy it as the output information of the composite filter
      this->GetOutput()->CopyInformation( m_Unwarp->GetOutput() );
    }
  else
    {
    m_TVDenoisingTime->SetInput(m_TVDenoisingSpace->GetOutput());

    // Have the last filter calculate its output information
    m_TVDenoisingTime->UpdateOutputInformation();

    // Copy it as the output information of the composite filter
    this->GetOutput()->CopyInformation( m_TVDenoisingTime->GetOutput() );
    }
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  // Declare the pointer that will be used to plug the output back as input
  typename VolumeSeriesType::Pointer pimg;

  for (int i=0; i<m_MainLoop_iterations; i++)
    {
    // After the first iteration, we need to use the output as input
    if (i>0)
      {
        if (m_PerformWarping)
          pimg = m_Unwarp->GetOutput();
        else
          pimg = m_TVDenoisingTime->GetOutput();

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

    if (m_PerformWarping)
      {
      m_WarpProbe.Start();
      m_Warp->Update();
      m_WarpProbe.Stop();
      }

    m_TVTimeProbe.Start();
    m_TVDenoisingTime->UpdateLargestPossibleRegion();
    m_TVTimeProbe.Stop();

    if (m_PerformWarping)
      {
      m_UnwarpProbe.Start();
      m_Unwarp->Update();
      m_UnwarpProbe.Stop();
      }
    }

  if (m_PerformWarping)
    this->GraftOutput( m_Unwarp->GetOutput() );
  else
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
  if (m_PerformWarping)
    {
    os << "  Warping volumes to average position: " << m_WarpProbe.GetTotal()
       << ' ' << m_WarpProbe.GetUnit() << std::endl;
    }
  os << "  Temporal total variation denoising: " << m_TVTimeProbe.GetTotal()
     << ' ' << m_TVTimeProbe.GetUnit() << std::endl;
  if (m_PerformWarping)
    {
    os << "  Warping corrections from average position: " << m_UnwarpProbe.GetTotal()
       << ' ' << m_UnwarpProbe.GetUnit() << std::endl;
    }
}

}// end namespace


#endif
