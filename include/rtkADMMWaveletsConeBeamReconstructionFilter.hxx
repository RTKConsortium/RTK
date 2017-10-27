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

#ifndef rtkADMMWaveletsConeBeamReconstructionFilter_hxx
#define rtkADMMWaveletsConeBeamReconstructionFilter_hxx

#include "rtkADMMWaveletsConeBeamReconstructionFilter.h"

namespace rtk
{

template< typename TOutputImage >
ADMMWaveletsConeBeamReconstructionFilter<TOutputImage>
::ADMMWaveletsConeBeamReconstructionFilter():
  m_Alpha(1),
  m_Beta(1),
  m_AL_iterations(10),
  m_CG_iterations(3),
  m_Order(3),
  m_NumberOfLevels(5)
{
  this->SetNumberOfRequiredInputs(2);

  // Create the filters
  m_ZeroMultiplyFilter = MultiplyFilterType::New();
  m_SubtractFilter1 = SubtractFilterType::New();
  m_SubtractFilter2 = SubtractFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_AddFilter1 = AddFilterType::New();
  m_AddFilter2 = AddFilterType::New();
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_SoftThresholdFilter = SoftThresholdFilterType::New();
  m_CGOperator = CGOperatorFilterType::New();
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();

  // Set permanent connections
  m_AddFilter1->SetInput2(m_ZeroMultiplyFilter->GetOutput());
  m_MultiplyFilter->SetInput1( m_AddFilter1->GetOutput() );
  m_AddFilter2->SetInput1(m_MultiplyFilter->GetOutput());
  m_ConjugateGradientFilter->SetB(m_AddFilter2->GetOutput());
  m_SubtractFilter1->SetInput1(m_ConjugateGradientFilter->GetOutput());
  m_SubtractFilter1->SetInput2(m_ZeroMultiplyFilter->GetOutput());
  m_SoftThresholdFilter->SetInput(m_SubtractFilter1->GetOutput());
  m_SubtractFilter2->SetInput1(m_SoftThresholdFilter->GetOutput());
  m_SubtractFilter2->SetInput2(m_SubtractFilter1->GetOutput());

  // Set permanent parameters
  m_ZeroMultiplyFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
  m_DisableDisplacedDetectorFilter = false;

  // Set memory management parameters
  m_ZeroMultiplyFilter->ReleaseDataFlagOn();
  m_AddFilter1->ReleaseDataFlagOn();
  m_AddFilter2->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_ConjugateGradientFilter->ReleaseDataFlagOff(); // Output is f_k+1
  m_SubtractFilter1->ReleaseDataFlagOff(); // Output used in two filters
  m_SoftThresholdFilter->ReleaseDataFlagOff(); // Output is g_k+1
  m_SubtractFilter2->ReleaseDataFlagOff(); //Output is d_k+1
  m_DisplacedDetectorFilter->ReleaseDataFlagOn();
}

template< typename TOutputImage >
void
ADMMWaveletsConeBeamReconstructionFilter<TOutputImage>
::SetForwardProjectionFilter (int _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_ForwardProjectionFilterForConjugateGradient = this->InstantiateForwardProjectionFilter( _arg );
    m_CGOperator->SetForwardProjectionFilter( m_ForwardProjectionFilterForConjugateGradient );
    }
}

template< typename TOutputImage >
void
ADMMWaveletsConeBeamReconstructionFilter<TOutputImage>
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

template< typename TOutputImage >
void
ADMMWaveletsConeBeamReconstructionFilter<TOutputImage>
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

template< typename TOutputImage >
void
ADMMWaveletsConeBeamReconstructionFilter<TOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_ZeroMultiplyFilter->SetInput1(this->GetInput(0));
  m_CGOperator->SetInput(1, this->GetInput(1)); // The projections (the conjugate gradient operator needs them)
  m_CGOperator->SetBeta(m_Beta);
  m_ConjugateGradientFilter->SetX(this->GetInput(0));
  m_MultiplyFilter->SetConstant2( m_Beta );
  m_DisplacedDetectorFilter->SetInput(this->GetInput(1));

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_BackProjectionFilter->SetInput(0, m_ZeroMultiplyFilter->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_DisplacedDetectorFilter->GetOutput());
  m_AddFilter1->SetInput1(this->GetInput(0));
  m_AddFilter2->SetInput2(m_BackProjectionFilter->GetOutput());

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_CG_iterations);
  m_SoftThresholdFilter->SetNumberOfLevels(this->GetNumberOfLevels());
  m_SoftThresholdFilter->SetOrder(this->GetOrder());
  m_SoftThresholdFilter->SetThreshold(m_Alpha/(2 * m_Beta));
  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);
  m_CGOperator->SetDisableDisplacedDetectorFilter(m_DisableDisplacedDetectorFilter);

  // Have the last filter calculate its output information
  m_SubtractFilter2->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_SubtractFilter2->GetOutput() );
}

template< typename TOutputImage >
void
ADMMWaveletsConeBeamReconstructionFilter<TOutputImage>
::GenerateData()
{
  typename TOutputImage::Pointer f_k_plus_one;
  typename TOutputImage::Pointer W_t_G_k_plus_one;
  typename TOutputImage::Pointer W_t_D_k_plus_one;

  for(unsigned int iter=0; iter < m_AL_iterations; iter++)
    {
    // After the first update, we need to use some outputs as inputs
    if(iter>0)
      {
      f_k_plus_one = m_ConjugateGradientFilter->GetOutput();
      f_k_plus_one->DisconnectPipeline();
      m_ConjugateGradientFilter->SetX(f_k_plus_one);

      W_t_G_k_plus_one = m_SoftThresholdFilter->GetOutput();
      W_t_G_k_plus_one->DisconnectPipeline();
      m_AddFilter1->SetInput1(W_t_G_k_plus_one);

      W_t_D_k_plus_one = m_SubtractFilter2->GetOutput();
      W_t_D_k_plus_one->DisconnectPipeline();
      m_AddFilter1->SetInput2(W_t_D_k_plus_one);
      m_SubtractFilter1->SetInput2(W_t_D_k_plus_one);

      // Recreate the links destroyed by DisconnectPipeline
      m_SubtractFilter1->SetInput1(m_ConjugateGradientFilter->GetOutput());
      m_SubtractFilter2->SetInput1(m_SoftThresholdFilter->GetOutput());
      }

    m_BeforeConjugateGradientProbe.Start();
    m_AddFilter2->Update();
    m_BeforeConjugateGradientProbe.Stop();

    m_ConjugateGradientProbe.Start();
    m_ConjugateGradientFilter->Update();
    m_ConjugateGradientProbe.Stop();

    m_WaveletsSoftTresholdingProbe.Start();
    m_SubtractFilter2->Update();
    m_WaveletsSoftTresholdingProbe.Stop();
    }
  this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
}

template< typename TOutputImage >
void
ADMMWaveletsConeBeamReconstructionFilter<TOutputImage>
::PrintTiming(std::ostream & os) const
{
  os << "ADMMWaveletsConeBeamReconstructionFilter timing:" << std::endl;
  os << "  Before conjugate gradient (computation of the B of AX=B): "
     << m_BeforeConjugateGradientProbe.GetTotal()
     << ' ' << m_BeforeConjugateGradientProbe.GetUnit() << std::endl;
  os << "  Conjugate gradient optimization: "
     << m_ConjugateGradientProbe.GetTotal()
     << ' ' << m_ConjugateGradientProbe.GetUnit() << std::endl;
  os << "  Wavelets soft thresholding: "
     << m_WaveletsSoftTresholdingProbe.GetTotal()
     << ' ' << m_WaveletsSoftTresholdingProbe.GetUnit() << std::endl;
}


}// end namespace


#endif
