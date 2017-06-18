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

#ifndef rtkADMMTotalVariationConjugateGradientOperator_hxx
#define rtkADMMTotalVariationConjugateGradientOperator_hxx

#include "rtkADMMTotalVariationConjugateGradientOperator.h"

namespace rtk
{

template< typename TOutputImage, typename TGradientOutputImage> 
ADMMTotalVariationConjugateGradientOperator<TOutputImage, TGradientOutputImage>
::ADMMTotalVariationConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(2);

  // Set default parameters
  m_Beta = 0;
  m_IsGated = false;

  // Create filters
  m_MultiplyFilter = MultiplyFilterType::New();
  m_ZeroMultiplyProjectionFilter = MultiplyFilterType::New();
  m_ZeroMultiplyVolumeFilter = MultiplyFilterType::New();
  m_SubtractFilter = SubtractFilterType::New();
  m_DivergenceFilter = DivergenceFilterType::New();
  m_GradientFilter = GradientFilterType::New();
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
  m_GatingWeightsFilter = GatingWeightsFilterType::New();

  // Set permanent connections
  m_DivergenceFilter->SetInput(m_GradientFilter->GetOutput());
  m_MultiplyFilter->SetInput1( m_DivergenceFilter->GetOutput() );
  m_SubtractFilter->SetInput2( m_MultiplyFilter->GetOutput() );

  // Set permanent parameters
  m_ZeroMultiplyProjectionFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_ZeroMultiplyVolumeFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
  m_DisableDisplacedDetectorFilter = false;

  // Set memory management options
  m_ZeroMultiplyProjectionFilter->ReleaseDataFlagOn();
  m_ZeroMultiplyVolumeFilter->ReleaseDataFlagOn();
  m_DivergenceFilter->ReleaseDataFlagOn();
  m_GradientFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_SubtractFilter->ReleaseDataFlagOff();
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConjugateGradientOperator<TOutputImage, TGradientOutputImage>
::SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg)
{
  if (m_BackProjectionFilter != _arg)
    this->Modified();
  m_BackProjectionFilter = _arg;
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConjugateGradientOperator<TOutputImage, TGradientOutputImage>
::SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg)
{
  if (m_ForwardProjectionFilter != _arg)
    this->Modified();
  m_ForwardProjectionFilter = _arg;
}

template< typename TOutputImage, typename TGradientOutputImage>
void
ADMMTotalVariationConjugateGradientOperator<TOutputImage, TGradientOutputImage>
::SetGatingWeights(std::vector<float> weights)
{
  m_GatingWeights = weights;
  m_IsGated = true;
  this->Modified();
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConjugateGradientOperator<TOutputImage, TGradientOutputImage>
::GenerateInputRequestedRegion()
{
  // Input 0 is the volume in which we backproject
  typename Superclass::InputImagePointer inputPtr0 =
    const_cast< TOutputImage * >( this->GetInput(0) );
  if ( !inputPtr0 ) return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the stack of projections to backproject
  typename Superclass::InputImagePointer  inputPtr1 =
    const_cast< TOutputImage * >( this->GetInput(1) );
  if ( !inputPtr1 ) return;
  inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
ADMMTotalVariationConjugateGradientOperator<TOutputImage, TGradientOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections, and connections with
  // forward and back projection filters, which are set
  // at runtime
  m_ForwardProjectionFilter->SetInput(0, m_ZeroMultiplyProjectionFilter->GetOutput());
  m_BackProjectionFilter->SetInput(0, m_ZeroMultiplyVolumeFilter->GetOutput());
  if(m_IsGated)
    {
    // Insert the gating filter into the pipeline
    m_GatingWeightsFilter->SetInput(m_ForwardProjectionFilter->GetOutput());
    m_GatingWeightsFilter->SetVector(m_GatingWeights);
    m_DisplacedDetectorFilter->SetInput(m_GatingWeightsFilter->GetOutput());
    }
  else
    {
    m_DisplacedDetectorFilter->SetInput(m_ForwardProjectionFilter->GetOutput());
    }
  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);
  m_BackProjectionFilter->SetInput(1, m_DisplacedDetectorFilter->GetOutput());
  m_SubtractFilter->SetInput1( m_BackProjectionFilter->GetOutput() );
  m_ZeroMultiplyVolumeFilter->SetInput1(this->GetInput(0));
  m_ZeroMultiplyProjectionFilter->SetInput1(this->GetInput(1));
  m_ForwardProjectionFilter->SetInput(1, this->GetInput(0));
  m_GradientFilter->SetInput(this->GetInput(0));

  // Set geometry
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());
  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);

  // Set runtime parameters
  m_MultiplyFilter->SetConstant2( m_Beta );

  // Set memory management parameters for forward
  // and back projection filters
  m_ForwardProjectionFilter->ReleaseDataFlagOn();
  m_BackProjectionFilter->ReleaseDataFlagOn();

  // Have the last filter calculate its output information
  m_SubtractFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_SubtractFilter->GetOutput() );
}

template< typename TOutputImage, typename TGradientOutputImage> 
void 
ADMMTotalVariationConjugateGradientOperator<TOutputImage, TGradientOutputImage>
::GenerateData()
{
    // Execute Pipeline
    m_SubtractFilter->Update();

    // Get the output
    this->GraftOutput( m_SubtractFilter->GetOutput() );
}

}// end namespace


#endif
