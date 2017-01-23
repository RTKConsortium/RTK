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

#ifndef rtkADMMTotalVariationConeBeamReconstructionFilter_hxx
#define rtkADMMTotalVariationConeBeamReconstructionFilter_hxx

#include "rtkADMMTotalVariationConeBeamReconstructionFilter.h"

namespace rtk
{

template< typename TOutputImage, typename TGradientOutputImage> 
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::ADMMTotalVariationConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_Alpha=1;
  m_Beta=1;
  m_AL_iterations=10;
  m_CG_iterations=3;
  m_IsGated=false;

  // Create the filters
  m_ZeroMultiplyVolumeFilter = MultiplyVolumeFilterType::New();
  m_ZeroMultiplyGradientFilter = MultiplyGradientFilterType::New();
  m_SubtractFilter1 = SubtractGradientsFilterType::New();
  m_SubtractFilter2 = SubtractGradientsFilterType::New();
  m_MultiplyFilter = MultiplyVolumeFilterType::New();
  m_GradientFilter1 = ImageGradientFilterType::New();
  m_GradientFilter2 = ImageGradientFilterType::New();
  m_SubtractVolumeFilter = SubtractVolumeFilterType::New();
  m_AddGradientsFilter = AddGradientsFilterType::New();
  m_DivergenceFilter = ImageDivergenceFilterType::New();
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_SoftThresholdFilter = SoftThresholdTVFilterType::New();
  m_CGOperator = CGOperatorFilterType::New();
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
  m_GatingWeightsFilter = GatingWeightsFilterType::New();

  // Set permanent connections
  m_ZeroMultiplyGradientFilter->SetInput1(m_GradientFilter1->GetOutput());
  m_AddGradientsFilter->SetInput1(m_ZeroMultiplyGradientFilter->GetOutput());
  m_AddGradientsFilter->SetInput2(m_GradientFilter1->GetOutput());
  m_DivergenceFilter->SetInput(m_AddGradientsFilter->GetOutput());
  m_MultiplyFilter->SetInput1( m_DivergenceFilter->GetOutput() );
  m_SubtractVolumeFilter->SetInput2(m_MultiplyFilter->GetOutput());
  m_ConjugateGradientFilter->SetB(m_SubtractVolumeFilter->GetOutput());
  m_ConjugateGradientFilter->SetNumberOfIterations(m_CG_iterations);
  m_GradientFilter2->SetInput(m_ConjugateGradientFilter->GetOutput());
  m_SubtractFilter1->SetInput1(m_GradientFilter2->GetOutput());
  m_SubtractFilter1->SetInput2(m_ZeroMultiplyGradientFilter->GetOutput());
  m_SoftThresholdFilter->SetInput(m_SubtractFilter1->GetOutput());
  m_SubtractFilter2->SetInput1(m_SoftThresholdFilter->GetOutput());
  m_SubtractFilter2->SetInput2(m_SubtractFilter1->GetOutput());

  // Set permanent parameters
  m_ZeroMultiplyVolumeFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_ZeroMultiplyGradientFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
  m_DisableDisplacedDetectorFilter = false;

  // Set memory management parameters
  m_ZeroMultiplyVolumeFilter->ReleaseDataFlagOn();
  m_ZeroMultiplyGradientFilter->ReleaseDataFlagOn();
  m_GradientFilter1->ReleaseDataFlagOn();
  m_AddGradientsFilter->ReleaseDataFlagOn();
  m_DivergenceFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_SubtractVolumeFilter->ReleaseDataFlagOn();
  m_ConjugateGradientFilter->ReleaseDataFlagOff(); // Output is f_k+1
  m_GradientFilter2->ReleaseDataFlagOn();
  m_SubtractFilter1->ReleaseDataFlagOff(); // Output used in two filters
  m_SoftThresholdFilter->ReleaseDataFlagOff(); // Output is g_k+1
  m_SubtractFilter2->ReleaseDataFlagOff(); //Output is d_k+1
  m_DisplacedDetectorFilter->ReleaseDataFlagOn();
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::SetForwardProjectionFilter (int _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter( _arg );
    m_CGOperator->SetForwardProjectionFilter( m_ForwardProjectionFilter );
    }
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::SetBackProjectionFilter (int _arg)
{
  if( _arg != this->GetBackProjectionFilter() )
    {
    Superclass::SetBackProjectionFilter( _arg );
    m_BackProjectionFilter = this->InstantiateBackProjectionFilter( _arg );
    m_BackProjectionFilterForConjugateGradient = this->InstantiateBackProjectionFilter( _arg );
    m_CGOperator->SetBackProjectionFilter( m_BackProjectionFilterForConjugateGradient );
    }
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::SetBetaForCurrentIteration(int iter)
{
  float currentBeta = m_Beta * (iter+1) / (float)m_AL_iterations;

  m_CGOperator->SetBeta(currentBeta);
  m_SoftThresholdFilter->SetThreshold(m_Alpha/(2 * currentBeta));
  m_MultiplyFilter->SetConstant2( (const float) currentBeta);
}

template< typename TOutputImage, typename TGradientOutputImage>
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::SetGatingWeights(std::vector<float> weights)
{
  m_GatingWeights = weights;
  m_IsGated = true;
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::GenerateInputRequestedRegion()
{
  // Input 0 is the volume we update
  typename Superclass::InputImagePointer inputPtr0 = const_cast< TOutputImage * >( this->GetInput(0) );
  if ( !inputPtr0 )
    {
    return;
    }
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the stack of projections to backproject
  typename Superclass::InputImagePointer  inputPtr1 = const_cast< TOutputImage * >( this->GetInput(1) );
  if ( !inputPtr1 )
    {
    return;
    }
  inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_GradientFilter1->SetInput(this->GetInput(0));
  m_ZeroMultiplyVolumeFilter->SetInput1(this->GetInput(0));
  m_CGOperator->SetInput(1, this->GetInput(1));
  m_ConjugateGradientFilter->SetX(this->GetInput(0));
  m_MultiplyFilter->SetConstant2( m_Beta );
  if (m_IsGated)
    {
    // Insert the gating filter into the pipeline
    m_GatingWeightsFilter->SetInput(this->GetInput(1));
    m_GatingWeightsFilter->SetVector(m_GatingWeights);
    m_DisplacedDetectorFilter->SetInput(m_GatingWeightsFilter->GetOutput());

    // Also perform gating in the conjugate gradient operator
    m_CGOperator->SetGatingWeights(m_GatingWeights);
    }
  else
    {
    m_DisplacedDetectorFilter->SetInput(this->GetInput(1));
    }
  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);
  m_CGOperator->SetDisableDisplacedDetectorFilter(m_DisableDisplacedDetectorFilter);

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_BackProjectionFilter->SetInput(0, m_ZeroMultiplyVolumeFilter->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_DisplacedDetectorFilter->GetOutput());
  m_SubtractVolumeFilter->SetInput1(m_BackProjectionFilter->GetOutput());

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_CG_iterations);

  // Have the last filter calculate its output information
  m_SubtractFilter2->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_SubtractFilter2->GetOutput() );
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::GenerateData()
{
  for(unsigned int iter=0; iter < m_AL_iterations; iter++)
    {
    SetBetaForCurrentIteration(iter);

    // After the first update, we need to use some outputs as inputs
    if(iter>0)
      {
      typename TOutputImage::Pointer f_k_plus_one = m_ConjugateGradientFilter->GetOutput();
      f_k_plus_one->DisconnectPipeline();
      m_ConjugateGradientFilter->SetX(f_k_plus_one);

      typename ImageGradientFilterType::OutputImageType::Pointer g_k_plus_one = m_SoftThresholdFilter->GetOutput();
      g_k_plus_one->DisconnectPipeline();
      m_AddGradientsFilter->SetInput2(g_k_plus_one);

      typename ImageGradientFilterType::OutputImageType::Pointer d_k_plus_one = m_SubtractFilter2->GetOutput();
      d_k_plus_one->DisconnectPipeline();
      m_AddGradientsFilter->SetInput1(d_k_plus_one);
      m_SubtractFilter1->SetInput2(d_k_plus_one);

      // Recreate the links destroyed by DisconnectPipeline
      m_GradientFilter2->SetInput(m_ConjugateGradientFilter->GetOutput());
      m_SubtractFilter2->SetInput1(m_SoftThresholdFilter->GetOutput());
      }

    m_BeforeConjugateGradientProbe.Start();
    m_SubtractVolumeFilter->Update();
    m_BeforeConjugateGradientProbe.Stop();

    m_ConjugateGradientProbe.Start();
    m_ConjugateGradientFilter->Update();
    m_ConjugateGradientProbe.Stop();

    m_TVSoftTresholdingProbe.Start();
    m_SubtractFilter2->Update();
    m_TVSoftTresholdingProbe.Stop();
    }
  this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
}

template< typename TOutputImage, typename TGradientOutputImage>
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>
::PrintTiming(std::ostream & os) const
{
  os << "ADMMWaveletsConeBeamReconstructionFilter timing:" << std::endl;
  os << "  Before conjugate gradient (computation of the B of AX=B): "
     << m_BeforeConjugateGradientProbe.GetTotal()
     << ' ' << m_BeforeConjugateGradientProbe.GetUnit() << std::endl;
  os << "  Conjugate gradient optimization: "
     << m_ConjugateGradientProbe.GetTotal()
     << ' ' << m_ConjugateGradientProbe.GetUnit() << std::endl;
  os << "  TV soft thresholding: "
     << m_TVSoftTresholdingProbe.GetTotal()
     << ' ' << m_TVSoftTresholdingProbe.GetUnit() << std::endl;
}

}// end namespace


#endif
