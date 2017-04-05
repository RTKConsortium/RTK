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
#ifndef rtkRegularizedConjugateGradientConeBeamReconstructionFilter_hxx
#define rtkRegularizedConjugateGradientConeBeamReconstructionFilter_hxx

#include "rtkRegularizedConjugateGradientConeBeamReconstructionFilter.h"

namespace rtk
{

template< typename TImage >
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>::RegularizedConjugateGradientConeBeamReconstructionFilter()
{
  // Set the default values of member parameters
  m_GammaTV = 0.00005;
  m_Gamma = 0; // Laplacian regularization
  m_SoftThresholdWavelets = 0.001;
  m_SoftThresholdOnImage = 0.001;
  m_Preconditioned = false;
  m_RegularizedCG = false;

  m_TV_iterations=10;
  m_MainLoop_iterations=10;
  m_CG_iterations=4;

  // Default pipeline: CG, positivity, spatial TV
  m_PerformPositivity = true;
  m_PerformTVSpatialDenoising = true;
  m_PerformWaveletsSpatialDenoising = false;
  m_PerformSoftThresholdOnImage= false;

  // Dimensions processed for TV, default is all
  for (unsigned int i=0; i<TImage::ImageDimension; i++)
    m_DimensionsProcessedForTV[i]=true;

  // Other parameters
  m_CudaConjugateGradient = true; // 3D volumes of usual size fit on GPUs
  m_Order = 5;
  m_NumberOfLevels = 3;
  m_DisableDisplacedDetectorFilter = false;
  m_IterationCosts = false;

  // Create the filters
  m_CGFilter = CGFilterType::New();
  m_PositivityFilter = ThresholdFilterType::New();
  m_TVDenoising = TVDenoisingFilterType::New();
  m_WaveletsDenoising = WaveletsDenoisingFilterType::New();
  m_SoftThresholdFilter = SoftThresholdFilterType::New();
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::SetInputVolume(const TImage* Volume)
{
  this->SetPrimaryInput(const_cast<TImage*>(Volume));
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::SetInputProjectionStack(const TImage* Projection)
{
  this->SetInput("ProjectionStack", const_cast<TImage*>(Projection));
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::SetInputWeights(const TImage* Weights)
{
  this->SetInput("Weights", const_cast<TImage*>(Weights));
}

template< typename TOutputImage>
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TOutputImage>::
SetSupportMask(const TOutputImage *SupportMask)
{
  this->SetInput("SupportMask", const_cast<TOutputImage*>(SupportMask));
}

template< typename TImage >
typename TImage::ConstPointer
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::GetInputVolume()
{
  return static_cast< const TImage * >
          ( this->itk::ProcessObject::GetInput("Primary") );
}

template< typename TImage >
typename TImage::Pointer
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::GetInputProjectionStack()
{
  return static_cast< TImage * >
          ( this->itk::ProcessObject::GetInput("ProjectionStack") );
}

template< typename TImage >
typename TImage::Pointer
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::GetInputWeights()
{
  return static_cast< TImage * >
          ( this->itk::ProcessObject::GetInput("Weights") );
}

template< typename TOutputImage>
typename TOutputImage::ConstPointer
RegularizedConjugateGradientConeBeamReconstructionFilter<TOutputImage>::
GetSupportMask()
{
  return static_cast< const TOutputImage * >
          ( this->itk::ProcessObject::GetInput("SupportMask") );
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::SetForwardProjectionFilter(int _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_CGFilter->SetForwardProjectionFilter( _arg );
    }
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::SetBackProjectionFilter(int _arg)
{
  if( _arg != this->GetBackProjectionFilter() )
    {
    Superclass::SetBackProjectionFilter( _arg );
    m_CGFilter->SetBackProjectionFilter( _arg );
    }
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Let the CG subfilters compute the requested regions for the projections
  // stack and the input volume
  m_CGFilter->PropagateRequestedRegion(m_CGFilter->GetOutput());
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::GenerateOutputInformation()
{
  // Construct the pipeline, adding regularization filters if the user wants them
  // Connect the last filter's output to the next filter's input using the currentDownstreamFilter pointer
  typename itk::ImageToImageFilter<TImage, TImage>::Pointer currentDownstreamFilter;

  // The conjugate gradient filter is the only part that must be in the pipeline
  // whatever was the user wants
  m_CGFilter->SetInput(0, this->GetInputVolume());
  m_CGFilter->SetInput(1, this->GetInputProjectionStack());
  m_CGFilter->SetInput(2, this->GetInputWeights());
  m_CGFilter->SetSupportMask(this->GetSupportMask());
  m_CGFilter->SetGeometry(this->m_Geometry);
  m_CGFilter->SetNumberOfIterations(this->m_CG_iterations);
  m_CGFilter->SetCudaConjugateGradient(this->GetCudaConjugateGradient());
  m_CGFilter->SetRegularized(this->m_RegularizedCG);
  m_CGFilter->SetGamma(this->m_Gamma);
  m_CGFilter->SetIterationCosts(m_IterationCosts);
  m_CGFilter->SetDisableDisplacedDetectorFilter(m_DisableDisplacedDetectorFilter);

  currentDownstreamFilter = m_CGFilter;

  // Plug the positivity filter if requested
  if (m_PerformPositivity)
    {
    m_PositivityFilter->SetInPlace(true);

    m_PositivityFilter->SetOutsideValue(0.0);
    m_PositivityFilter->ThresholdBelow(0.0);
    m_PositivityFilter->SetInput(currentDownstreamFilter->GetOutput());

    currentDownstreamFilter = m_PositivityFilter;
    }

  if (m_PerformTVSpatialDenoising)
    {
    currentDownstreamFilter->ReleaseDataFlagOn();

    m_TVDenoising->SetInput(currentDownstreamFilter->GetOutput());
    m_TVDenoising->SetNumberOfIterations(this->m_TV_iterations);
    m_TVDenoising->SetGamma(this->m_GammaTV);
    m_TVDenoising->SetDimensionsProcessed(this->m_DimensionsProcessedForTV);

    currentDownstreamFilter = m_TVDenoising;
    }

  if (m_PerformWaveletsSpatialDenoising)
    {
    currentDownstreamFilter->ReleaseDataFlagOn();

    m_WaveletsDenoising->SetInput(currentDownstreamFilter->GetOutput());
    m_WaveletsDenoising->SetOrder(m_Order);
    m_WaveletsDenoising->SetThreshold(m_SoftThresholdWavelets);
    m_WaveletsDenoising->SetNumberOfLevels(m_NumberOfLevels);

    currentDownstreamFilter = m_WaveletsDenoising;
    }

  if (m_PerformSoftThresholdOnImage)
    {
    currentDownstreamFilter->ReleaseDataFlagOn();

    m_SoftThresholdFilter->SetInput(currentDownstreamFilter->GetOutput());
    m_SoftThresholdFilter->SetThreshold(m_SoftThresholdOnImage);

    currentDownstreamFilter = m_SoftThresholdFilter;
    }

  // Have the last filter calculate its output information
  currentDownstreamFilter->ReleaseDataFlagOff();
  currentDownstreamFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( currentDownstreamFilter->GetOutput() );
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::GenerateData()
{
  // Declare the pointer that will be used to plug the output back as input
  typename itk::ImageToImageFilter<TImage, TImage>::Pointer currentDownstreamFilter;
  typename TImage::Pointer pimg;

  for (int i=0; i<m_MainLoop_iterations; i++)
    {
    // After the first iteration, we need to use the output as input
    if (i>0)
      {
      pimg = currentDownstreamFilter->GetOutput();

      pimg->DisconnectPipeline();
      m_CGFilter->SetInput(0, pimg);

      // The input volume is no longer needed on the GPU, so we transfer it back to the CPU
      this->GetInputVolume()->GetBufferPointer();
      }

    m_CGProbe.Start();
    m_CGFilter->Update();
    m_CGProbe.Stop();
    currentDownstreamFilter = m_CGFilter;

    if (m_PerformPositivity)
      {
      m_PositivityProbe.Start();
      m_PositivityFilter->Update();
      m_PositivityProbe.Stop();

      currentDownstreamFilter = m_PositivityFilter;
      }

    if (m_PerformTVSpatialDenoising)
      {
      m_TVSpatialDenoisingProbe.Start();
      m_TVDenoising->Update();
      m_TVSpatialDenoisingProbe.Stop();

      currentDownstreamFilter = m_TVDenoising;
      }

    if (m_PerformWaveletsSpatialDenoising)
      {
      m_WaveletsSpatialDenoisingProbe.Start();
      m_WaveletsDenoising->Update();
      m_WaveletsSpatialDenoisingProbe.Stop();

      currentDownstreamFilter = m_WaveletsDenoising;
      }

    if (m_PerformSoftThresholdOnImage)
      {
      m_SoftThresholdImageProbe.Start();
      m_SoftThresholdFilter->Update();
      m_SoftThresholdImageProbe.Stop();

      currentDownstreamFilter = m_SoftThresholdFilter;
      }
    }

  this->GraftOutput( currentDownstreamFilter->GetOutput() );
}

template< typename TImage >
void
RegularizedConjugateGradientConeBeamReconstructionFilter<TImage>
::PrintTiming(std::ostream& os) const
{
  os << "RegularizedConjugateGradientConeBeamReconstructionFilter timing:" << std::endl;
  os << "  3D conjugate gradient reconstruction: " << m_CGProbe.GetTotal()
     << ' ' << m_CGProbe.GetUnit() << std::endl;
  if (m_PerformPositivity)
    {
    os << "  Positivity enforcement: " << m_PositivityProbe.GetTotal()
      << ' ' << m_PositivityProbe.GetUnit() << std::endl;
    }
  if (m_PerformTVSpatialDenoising)
    {
    os << "  Total Variation spatial denoising: " << m_TVSpatialDenoisingProbe.GetTotal()
      << ' ' << m_TVSpatialDenoisingProbe.GetUnit() << std::endl;
    }
  if (m_PerformWaveletsSpatialDenoising)
    {
    os << "  Wavelets spatial denoising: " << m_WaveletsSpatialDenoisingProbe.GetTotal()
      << ' ' << m_WaveletsSpatialDenoisingProbe.GetUnit() << std::endl;
    }
  if (m_PerformSoftThresholdOnImage)
    {
    os << "  Soft thresholding in image domain: " << m_SoftThresholdImageProbe.GetTotal()
      << ' ' << m_SoftThresholdImageProbe.GetUnit() << std::endl;
    }
}

}// end namespace


#endif
