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

#ifndef __rtkSIRTConeBeamReconstructionFilter_txx
#define __rtkSIRTConeBeamReconstructionFilter_txx

#include "rtkSIRTConeBeamReconstructionFilter.h"

namespace rtk
{

template< typename TOutputImage>
SIRTConeBeamReconstructionFilter<TOutputImage>::SIRTConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_NumberOfIterations=3;
  m_MeasureExecutionTimes=false;
  m_CurrentBackProjectionConfiguration = -1;
  m_CurrentForwardProjectionConfiguration = -1;

  // Create the filters
  m_ZeroMultiplyVolumeFilter = MultiplyVolumeFilterType::New();
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_CGOperator = CGOperatorFilterType::New();
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());

  // Set permanent parameters
  m_ZeroMultiplyVolumeFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());

  // Set memory management parameters
  m_ZeroMultiplyVolumeFilter->ReleaseDataFlagOn();
}

template< typename TOutputImage>
void
SIRTConeBeamReconstructionFilter<TOutputImage>
::ConfigureForwardProjection (int _arg)
{
  switch(_arg)
    {
    case(0):
      m_ForwardProjectionFilter = rtk::JosephForwardProjectionImageFilter<TOutputImage, TOutputImage>::New();
    break;
    case(1):
    #if CUDA_FOUND
      m_ForwardProjectionFilter = rtk::CudaForwardProjectionImageFilter::New();
    #else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
    #endif
    break;
    case(2):
      m_ForwardProjectionFilter = rtk::RayCastInterpolatorForwardProjectionImageFilter<TOutputImage, TOutputImage>::New();
    break;

    default:
      std::cerr << "Unhandled --method value." << std::endl;
    }
  m_CGOperator->SetForwardProjectionFilter( m_ForwardProjectionFilter );

  if (m_CurrentForwardProjectionConfiguration != _arg)
    {
    this->Modified();
    m_CGOperator->Modified();
    }
}

template< typename TOutputImage>
void
SIRTConeBeamReconstructionFilter<TOutputImage>
::ConfigureBackProjection (int _arg)
{
  switch(_arg)
    {
    case(0):
      m_BackProjectionFilterForConjugateGradient  = rtk::BackProjectionImageFilter<TOutputImage, TOutputImage>::New();
      m_BackProjectionFilter = rtk::BackProjectionImageFilter<TOutputImage, TOutputImage>::New();
      break;
    case(1):
      m_BackProjectionFilterForConjugateGradient = rtk::JosephBackProjectionImageFilter<TOutputImage, TOutputImage>::New();
      m_BackProjectionFilter = rtk::JosephBackProjectionImageFilter<TOutputImage, TOutputImage>::New();
      break;
    case(2):
    #if CUDA_FOUND
      m_BackProjectionFilterForConjugateGradient = rtk::CudaBackProjectionImageFilter::New();
      m_BackProjectionFilter = rtk::CudaBackProjectionImageFilter::New();
    #else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
    #endif
    break;

    default:
      std::cerr << "Unhandled --bp value." << std::endl;
    }
  m_CGOperator->SetBackProjectionFilter( m_BackProjectionFilterForConjugateGradient );

  if (m_CurrentBackProjectionConfiguration != _arg)
    {
    this->Modified();
    m_CGOperator->Modified();
    }
}

template< typename TOutputImage>
void
SIRTConeBeamReconstructionFilter<TOutputImage>
::GenerateInputRequestedRegion()
{
    // Input 0 is the volume we update
    typename Superclass::InputImagePointer inputPtr0 =
            const_cast< TOutputImage * >( this->GetInput(0) );
    if ( !inputPtr0 )
        return;
    inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

    // Input 1 is the stack of projections to backproject
    typename Superclass::InputImagePointer  inputPtr1 =
            const_cast< TOutputImage * >( this->GetInput(1) );
    if ( !inputPtr1 )
        return;
    inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );

}

template< typename TOutputImage>
void
SIRTConeBeamReconstructionFilter<TOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_ZeroMultiplyVolumeFilter->SetInput1(this->GetInput(0));
  m_CGOperator->SetInput(1, this->GetInput(1));
  m_ConjugateGradientFilter->SetX(this->GetInput(0));

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_BackProjectionFilter->SetInput(0, m_ZeroMultiplyVolumeFilter->GetOutput());
  m_BackProjectionFilter->SetInput(1, this->GetInput(1));
  m_ConjugateGradientFilter->SetB(m_BackProjectionFilter->GetOutput());

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);

  // Have the last filter calculate its output information
  m_ConjugateGradientFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_ConjugateGradientFilter->GetOutput() );
}

template< typename TOutputImage>
void
SIRTConeBeamReconstructionFilter<TOutputImage>
::GenerateData()
{
  itk::TimeProbe SIRTTimeProbe;

  if(m_MeasureExecutionTimes)
    {
    std::cout << "Starting SIRT" << std::endl;
    SIRTTimeProbe.Start();
    }

  m_ConjugateGradientFilter->Update();

  if(m_MeasureExecutionTimes)
    {
    SIRTTimeProbe.Stop();
    std::cout << "SIRT took " << SIRTTimeProbe.GetTotal() << ' ' << SIRTTimeProbe.GetUnit() << std::endl;
    }

  this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
}

}// end namespace


#endif
