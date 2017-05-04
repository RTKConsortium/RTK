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
#ifndef rtkFourDROOSTERConeBeamReconstructionFilter_hxx
#define rtkFourDROOSTERConeBeamReconstructionFilter_hxx

#include "rtkFourDROOSTERConeBeamReconstructionFilter.h"
#include <itkImageFileWriter.h>

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType>
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::FourDROOSTERConeBeamReconstructionFilter()
{
//   this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_GammaTVSpace = 0.00005;
  m_GammaTVTime = 0.0002;
  m_GammaTNV = 0.0002;
  m_LambdaL0Time = 0.005;
  m_SoftThresholdWavelets = 0.001;
  
  m_TV_iterations=10;
  m_MainLoop_iterations=10;
  m_CG_iterations=4;
  m_L0_iterations=5;
  
  // Default pipeline: 4DCG, positivity, motion mask, spatial TV, temporal TV
  m_PerformPositivity = true;
  m_PerformMotionMask = true;
  m_PerformTVSpatialDenoising = true;
  m_PerformWaveletsSpatialDenoising = false;
  m_PerformWarping = false;
  m_PerformTVTemporalDenoising = true;
  m_PerformL0TemporalDenoising = false;
  m_PerformTNVDenoising = false;
  m_ComputeInverseWarpingByConjugateGradient=true;

  // Other parameters
  m_UseNearestNeighborInterpolationInWarping=false;
  m_PhaseShift = 0;
  m_CudaConjugateGradient = false; // 4D volumes of usual size only fit on the largest GPUs
  m_UseCudaCyclicDeformation = false;
  m_Order = 5;
  m_NumberOfLevels = 3;
  m_DisableDisplacedDetectorFilter = false;

  // Create the filters
  m_FourDCGFilter = FourDCGFilterType::New();
  m_PositivityFilter = ThresholdFilterType::New();
  m_ResampleFilter = ResampleFilterType::New();
  m_AverageOutOfROIFilter = AverageOutOfROIFilterType::New();
  m_TVDenoisingTime = TemporalTVDenoisingFilterType::New();
#ifdef RTK_USE_CUDA
  m_AverageOutOfROIFilter = rtk::CudaAverageOutOfROIImageFilter::New();
  m_TVDenoisingTime = rtk::CudaLastDimensionTVDenoisingImageFilter::New();
#endif
  m_TVDenoisingSpace = SpatialTVDenoisingFilterType::New();
  m_WaveletsDenoisingSpace = SpatialWaveletsDenoisingFilterType::New();
  m_Warp = WarpSequenceFilterType::New();
  m_Unwarp = UnwarpSequenceFilterType::New();
  m_InverseWarp = WarpSequenceFilterType::New();
  m_SubtractFilter = SubtractFilterType::New();
  m_AddFilter = AddFilterType::New();
  m_L0DenoisingTime = TemporalL0DenoisingFilterType::New();
  m_TNVDenoising = TNVDenoisingFilterType::New();

  // Initialize downstream filter
  m_DownstreamFilter = m_FourDCGFilter;
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetPrimaryInput(const_cast<VolumeSeriesType*>(VolumeSeries));
//   this->SetPrimaryInputName("VolumeSeries");
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetInput("ProjectionStack", const_cast<ProjectionStackType*>(Projection));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetMotionMask(const VolumeType* mask)
{
  this->SetInput("MotionMask", const_cast<VolumeType*>(mask));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetDisplacementField(const DVFSequenceImageType* DVFs)
{
  this->SetInput("DisplacementField", const_cast<DVFSequenceImageType*>(DVFs));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetInverseDisplacementField(const DVFSequenceImageType* DVFs)
{
  this->SetInput("InverseDisplacementField", const_cast<DVFSequenceImageType*>(DVFs));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename VolumeSeriesType::ConstPointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput("Primary") );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename ProjectionStackType::Pointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetInputProjectionStack()
{
  return static_cast< ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput("ProjectionStack") );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::VolumeType::Pointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetMotionMask()
{
  return static_cast< VolumeType * >
          ( this->itk::ProcessObject::GetInput("MotionMask") );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::DVFSequenceImageType::Pointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetDisplacementField()
{
  return static_cast< DVFSequenceImageType * >
          ( this->itk::ProcessObject::GetInput("DisplacementField") );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::DVFSequenceImageType::Pointer
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetInverseDisplacementField()
{
  return static_cast< DVFSequenceImageType * >
          ( this->itk::ProcessObject::GetInput("InverseDisplacementField") );
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
::SetSignal(const std::vector<double> signal)
{
  m_FourDCGFilter->SetSignal(signal);
  this->m_Signal = signal;
  this->Modified();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Let the 4DCG subfilters compute the requested regions for the projections
  // stack and the input volume
  m_FourDCGFilter->PropagateRequestedRegion(m_FourDCGFilter->GetOutput());

  // Set the requested region to the full input for all regularization steps
  if (m_PerformMotionMask)
    {
    typename VolumeType::Pointer motionMaskPtr = this->GetMotionMask();
    motionMaskPtr->SetRequestedRegionToLargestPossibleRegion();
    }
  else
    {
    this->RemoveInput("MotionMask");
    }
  
  if (m_PerformWarping)
    {
    typename DVFSequenceImageType::Pointer DisplacementFieldPtr  = this->GetDisplacementField();
    DisplacementFieldPtr->SetRequestedRegionToLargestPossibleRegion();

    if (!m_ComputeInverseWarpingByConjugateGradient)
      {
      typename DVFSequenceImageType::Pointer InverseDisplacementFieldPtr  = this->GetInverseDisplacementField();
      InverseDisplacementFieldPtr->SetRequestedRegionToLargestPossibleRegion();
      }
    else
      {
      this->RemoveInput("InverseDisplacementField");
      }
    }
  else
    {
    // If the filter is used with m_PerformWarping = true, and then with
    // m_PerformWarping = false, it keeps requesting a region of the
    // input DVF, which by default may be larger than the largest
    // possible region of the DVF (it is the largest possible region of
    // the first input, and the sizes do not necessarily match).
    // This occurs, for example, in the fourdroostercudatest.
    this->RemoveInput("DisplacementField");
    this->RemoveInput("InverseDisplacementField");
    }
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  const int Dimension = VolumeType::ImageDimension;

  // The 4D conjugate gradient filter is the only part that must be in the pipeline
  // whatever was the user wants
  m_FourDCGFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_FourDCGFilter->SetInputProjectionStack(this->GetInputProjectionStack());
  m_FourDCGFilter->SetGeometry(this->m_Geometry);
  m_FourDCGFilter->SetNumberOfIterations(this->m_CG_iterations);
  m_FourDCGFilter->SetCudaConjugateGradient(this->GetCudaConjugateGradient());
  m_FourDCGFilter->SetDisableDisplacedDetectorFilter(m_DisableDisplacedDetectorFilter);
  m_DownstreamFilter = m_FourDCGFilter;
  
  // Plug the positivity filter if requested
  if (m_PerformPositivity)
    {
    m_PositivityFilter->SetInPlace(true);  

    m_PositivityFilter->SetOutsideValue(0.0);
    m_PositivityFilter->ThresholdBelow(0.0);
    m_PositivityFilter->SetInput(m_DownstreamFilter->GetOutput());
  
    m_DownstreamFilter = m_PositivityFilter;
    }
  
  // Etc..
  if (m_PerformMotionMask)
    {
    m_AverageOutOfROIFilter->SetInPlace(true);  
      
    m_AverageOutOfROIFilter->SetInput(m_DownstreamFilter->GetOutput());
    m_ResampleFilter->SetInput(this->GetMotionMask());
    m_AverageOutOfROIFilter->SetROI(m_ResampleFilter->GetOutput());

    // Set the resample filter
    typedef itk::IdentityTransform<double, Dimension> TransformType;
    typedef itk::NearestNeighborInterpolateImageFunction<VolumeType, double> InterpolatorType;
    m_ResampleFilter->SetInterpolator(InterpolatorType::New());
    m_ResampleFilter->SetTransform(TransformType::New());
    typename VolumeType::SizeType VolumeSize;
    typename VolumeType::SpacingType VolumeSpacing;
    typename VolumeType::PointType VolumeOrigin;
    typename VolumeType::DirectionType VolumeDirection;
    VolumeSize.Fill(0);
    VolumeSpacing.Fill(0);
    VolumeOrigin.Fill(0);
    VolumeDirection.Fill(0);
    for (int i=0; i<Dimension; i++)
      {
      VolumeSize[i] = this->GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize()[i];
      VolumeSpacing[i] = this->GetInputVolumeSeries()->GetSpacing()[i];
      VolumeOrigin[i] = this->GetInputVolumeSeries()->GetOrigin()[i];
      for (int j=0; j<Dimension; j++)
        {
        VolumeDirection(i,j) = this->GetInputVolumeSeries()->GetDirection()(i,j);
        }
      }
    m_ResampleFilter->SetSize(VolumeSize);
    m_ResampleFilter->SetOutputSpacing(VolumeSpacing);
    m_ResampleFilter->SetOutputOrigin(VolumeOrigin);
    m_ResampleFilter->SetOutputDirection(VolumeDirection);
    
    m_DownstreamFilter = m_AverageOutOfROIFilter;
    }
    
  if (m_PerformTVSpatialDenoising)
    {
    m_DownstreamFilter->ReleaseDataFlagOn();

    m_TVDenoisingSpace->SetInput(m_DownstreamFilter->GetOutput());
    m_TVDenoisingSpace->SetNumberOfIterations(this->m_TV_iterations);
    m_TVDenoisingSpace->SetGamma(this->m_GammaTVSpace);
    m_DimensionsProcessedForTVSpace[0]=true;
    m_DimensionsProcessedForTVSpace[1]=true;
    m_DimensionsProcessedForTVSpace[2]=true;
    m_DimensionsProcessedForTVSpace[3]=false;
    m_TVDenoisingSpace->SetDimensionsProcessed(this->m_DimensionsProcessedForTVSpace);
  
    m_DownstreamFilter = m_TVDenoisingSpace;
    }
    
  if (m_PerformWaveletsSpatialDenoising)
    {
    m_DownstreamFilter->ReleaseDataFlagOn();
      
    m_WaveletsDenoisingSpace->SetInput(m_DownstreamFilter->GetOutput());
    m_WaveletsDenoisingSpace->SetOrder(m_Order);
    m_WaveletsDenoisingSpace->SetThreshold(m_SoftThresholdWavelets);
    m_WaveletsDenoisingSpace->SetNumberOfLevels(m_NumberOfLevels);
  
    m_DownstreamFilter = m_WaveletsDenoisingSpace;
    }

  if (m_PerformWarping)
    {
    m_DownstreamFilter->ReleaseDataFlagOff();
      
    m_Warp->SetInput(m_DownstreamFilter->GetOutput());
    m_Warp->SetDisplacementField(this->GetDisplacementField());
    m_Warp->SetPhaseShift(m_PhaseShift);
    m_Warp->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);
    m_Warp->SetUseCudaCyclicDeformation(m_UseCudaCyclicDeformation);
  
    m_DownstreamFilter = m_Warp;
    }
    
  if (m_PerformTVTemporalDenoising)
    {
    m_DownstreamFilter->ReleaseDataFlagOff();
  
    if (m_PerformWarping && !m_ComputeInverseWarpingByConjugateGradient)
      m_TVDenoisingTime->SetInPlace(false);
    else
      m_TVDenoisingTime->SetInPlace(true);

    m_TVDenoisingTime->SetInput(m_DownstreamFilter->GetOutput());
    m_TVDenoisingTime->SetNumberOfIterations(this->m_TV_iterations);
    m_TVDenoisingTime->SetGamma(this->m_GammaTVTime);
    m_DimensionsProcessedForTVTime[0]=false;
    m_DimensionsProcessedForTVTime[1]=false;
    m_DimensionsProcessedForTVTime[2]=false;
    m_DimensionsProcessedForTVTime[3]=true;
    m_TVDenoisingTime->SetDimensionsProcessed(this->m_DimensionsProcessedForTVTime);
    m_TVDenoisingTime->SetBoundaryConditionToPeriodic();
    
    m_DownstreamFilter = m_TVDenoisingTime;
    }
    
  if (m_PerformL0TemporalDenoising)
    {
    m_DownstreamFilter->ReleaseDataFlagOff();

    if (m_PerformWarping && !m_ComputeInverseWarpingByConjugateGradient && !m_PerformTVTemporalDenoising)
      m_L0DenoisingTime->SetInPlace(false);
    else
      m_L0DenoisingTime->SetInPlace(true);
      
    m_L0DenoisingTime->SetInPlace(false);
      
    m_L0DenoisingTime->SetInput(m_DownstreamFilter->GetOutput());
    m_L0DenoisingTime->SetNumberOfIterations(this->m_L0_iterations);
    m_L0DenoisingTime->SetLambda(this->m_LambdaL0Time);
  
    m_DownstreamFilter = m_L0DenoisingTime;
    }
    
  if (m_PerformTNVDenoising)
    {
    m_DownstreamFilter->ReleaseDataFlagOff();

    if (m_PerformWarping && !m_ComputeInverseWarpingByConjugateGradient && !m_PerformTVTemporalDenoising && !m_PerformL0TemporalDenoising)
      m_TNVDenoising->SetInPlace(false);
    else
      m_TNVDenoising->SetInPlace(true);

    m_TNVDenoising->SetInput(m_DownstreamFilter->GetOutput());
    m_TNVDenoising->SetNumberOfIterations(this->m_TV_iterations);
    m_TNVDenoising->SetGamma(this->m_GammaTNV);

    m_DownstreamFilter = m_TNVDenoising;
    }

  if (m_PerformWarping)
    {
    if (m_ComputeInverseWarpingByConjugateGradient)
      {
      m_DownstreamFilter->ReleaseDataFlagOn();

      m_Unwarp->SetNumberOfIterations(4);
      m_Unwarp->SetInput(m_DownstreamFilter->GetOutput());
      m_Unwarp->SetDisplacementField(this->GetDisplacementField());
      m_Unwarp->SetPhaseShift(m_PhaseShift);
      m_Unwarp->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);
      m_Unwarp->SetCudaConjugateGradient(this->GetCudaConjugateGradient());
      m_Unwarp->SetUseCudaCyclicDeformation(m_UseCudaCyclicDeformation);

      m_DownstreamFilter = m_Unwarp;
      }
    else
      {
      m_DownstreamFilter->ReleaseDataFlagOff();
        
      // Compute the correction performed by temporal TV and/or temporal L0 and/or TNV
      m_SubtractFilter->SetInput1(m_DownstreamFilter->GetOutput());
      m_SubtractFilter->SetInput2(m_Warp->GetOutput());

      // Deform only that correction with the inverse field
      m_InverseWarp->SetInput(0, m_SubtractFilter->GetOutput());
      m_InverseWarp->SetDisplacementField(this->GetInverseDisplacementField());
      m_InverseWarp->SetPhaseShift(m_PhaseShift);
      m_InverseWarp->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);
      m_InverseWarp->SetUseCudaCyclicDeformation(m_UseCudaCyclicDeformation);

      // Add the deformed correction to the spatially denoised image to get the output
      m_AddFilter->SetInput1(m_InverseWarp->GetOutput());
      m_AddFilter->SetInput2(m_Warp->GetInput());

      m_DownstreamFilter = m_AddFilter;
      
      m_Warp->ReleaseDataFlagOff();
      m_SubtractFilter->ReleaseDataFlagOn();
      m_InverseWarp->ReleaseDataFlagOn();
      m_AddFilter->ReleaseDataFlagOff();
      }
    }
    
  // Have the last filter calculate its output information
  m_DownstreamFilter->ReleaseDataFlagOff();
  m_DownstreamFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_DownstreamFilter->GetOutput() );
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
      pimg = m_DownstreamFilter->GetOutput();

      pimg->DisconnectPipeline();
      m_FourDCGFilter->SetInputVolumeSeries(pimg);

      // The input volume is no longer needed on the GPU, so we transfer it back to the CPU
      this->GetInputVolumeSeries()->GetBufferPointer();
      }

    m_DownstreamFilter->Update();
    }

  this->GraftOutput( m_DownstreamFilter->GetOutput() );
}

//template< typename VolumeSeriesType, typename ProjectionStackType>
//void
//FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
//::PrintTiming(std::ostream& os) const
//{
//  os << "FourDROOSTERConeBeamReconstructionFilter timing:" << std::endl;
//  os << "  4D conjugate gradient reconstruction: " << m_CGProbe.GetTotal()
//     << ' ' << m_CGProbe.GetUnit() << std::endl;
//  if (m_PerformPositivity)
//    {
//    os << "  Positivity enforcement: " << m_PositivityProbe.GetTotal()
//      << ' ' << m_PositivityProbe.GetUnit() << std::endl;
//    }
//  if (m_PerformMotionMask)
//    {
//    os << "  Averaging along time outside the ROI where movement is allowed: " << m_MotionMaskProbe.GetTotal()
//      << ' ' << m_MotionMaskProbe.GetUnit() << std::endl;
//    }
//  if (m_PerformTVSpatialDenoising)
//    {
//    os << "  Total Variation spatial denoising: " << m_TVSpatialDenoisingProbe.GetTotal()
//      << ' ' << m_TVSpatialDenoisingProbe.GetUnit() << std::endl;
//    }
//  if (m_PerformWaveletsSpatialDenoising)
//    {
//    os << "  Wavelets spatial denoising: " << m_WaveletsSpatialDenoisingProbe.GetTotal()
//      << ' ' << m_WaveletsSpatialDenoisingProbe.GetUnit() << std::endl;
//    }
//  if (m_PerformWarping)
//    {
//    os << "  Warping volumes to reference position: " << m_WarpingProbe.GetTotal()
//       << ' ' << m_WarpingProbe.GetUnit() << std::endl;
//    }
//  if (m_PerformTVTemporalDenoising)
//    {
//    os << "  Temporal total variation denoising: " << m_TVTemporalDenoisingProbe.GetTotal()
//      << ' ' << m_TVTemporalDenoisingProbe.GetUnit() << std::endl;
//    }
//  if (m_PerformL0TemporalDenoising)
//    {
//    os << "  Gradient's L0 norm temporal denoising: " << m_L0TemporalDenoisingProbe.GetTotal()
//      << ' ' << m_L0TemporalDenoisingProbe.GetUnit() << std::endl;
//    }
//  if (m_PerformWarping)
//    {
//    os << "  Warping volumes back from average position: " << m_UnwarpingProbe.GetTotal()
//       << ' ' << m_UnwarpingProbe.GetUnit() << std::endl;
//    }
//}

}// end namespace


#endif
