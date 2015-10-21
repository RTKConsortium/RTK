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

#ifndef __rtkConjugateGradientConeBeamReconstructionFilter_txx
#define __rtkConjugateGradientConeBeamReconstructionFilter_txx

#include "rtkConjugateGradientConeBeamReconstructionFilter.h"

namespace rtk
{

template< typename TOutputImage>
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>::ConjugateGradientConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_NumberOfIterations=3;
  m_MeasureExecutionTimes=false;
  m_IsWeighted=false;

  // Create the filters
#ifdef RTK_USE_CUDA
  m_ConjugateGradientFilter = rtk::CudaConjugateGradientImageFilter_3f::New();
  m_DisplacedDetectorFilter = rtk::CudaDisplacedDetectorImageFilter::New();
  m_ConstantImageSource     = rtk::CudaConstantVolumeSource::New();
#else
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
  m_ConstantImageSource     = ConstantImageSourceType::New();
#endif
  m_CGOperator = CGOperatorFilterType::New();
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());

  // Set permanent parameters
  m_ConstantImageSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);

  // Set memory management parameters
  m_ConstantImageSource->ReleaseDataFlagOn();
}

template< typename TOutputImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>
::SetForwardProjectionFilter (int _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter( _arg );
    m_CGOperator->SetForwardProjectionFilter( m_ForwardProjectionFilter );
    }
}

template< typename TOutputImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>
::SetBackProjectionFilter (int _arg)
{
  if( _arg != this->GetBackProjectionFilter() )
    {
    Superclass::SetBackProjectionFilter( _arg );
    m_BackProjectionFilter = this->InstantiateBackProjectionFilter( _arg );
    m_BackProjectionFilterForB = this->InstantiateBackProjectionFilter( _arg );
    m_CGOperator->SetBackProjectionFilter( m_BackProjectionFilter);
    }
}

template< typename TOutputImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>
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

  if (m_IsWeighted)
    {
    this->SetNumberOfRequiredInputs(3);

    // Input 2 is the weights map, if any
    typename Superclass::InputImagePointer  inputPtr2 =
            const_cast< TOutputImage * >( this->GetInput(2) );
    if ( !inputPtr2 )
        return;
    inputPtr2->SetRequestedRegion( inputPtr2->GetLargestPossibleRegion() );
    }
}

template< typename TOutputImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_ConstantImageSource->SetInformationFromImage(this->GetInput(0));
  m_CGOperator->SetInput(1, this->GetInput(1));
  m_ConjugateGradientFilter->SetX(this->GetInput(0));
  m_DisplacedDetectorFilter->SetInput(this->GetInput(1));

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_BackProjectionFilterForB->SetInput(0, m_ConstantImageSource->GetOutput());
  m_ConjugateGradientFilter->SetB(m_BackProjectionFilterForB->GetOutput());
  if (m_IsWeighted)
    {
    m_MultiplyFilter = MultiplyFilterType::New();
    m_MultiplyFilter->SetInput1(m_DisplacedDetectorFilter->GetOutput());
    m_MultiplyFilter->SetInput2(this->GetInput(2));
    m_CGOperator->SetInput(2, this->GetInput(2));
    m_BackProjectionFilterForB->SetInput(1, m_MultiplyFilter->GetOutput());
    }
  else
    {
    m_BackProjectionFilterForB->SetInput(1, m_DisplacedDetectorFilter->GetOutput());
    }

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->m_Geometry);
  m_BackProjectionFilterForB->SetGeometry(this->m_Geometry.GetPointer());
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);
  m_CGOperator->SetIsWeighted(m_IsWeighted);

  // Have the last filter calculate its output information
  m_ConjugateGradientFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_ConjugateGradientFilter->GetOutput() );
}

template< typename TOutputImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>
::GenerateData()
{
  itk::TimeProbe ConjugateGradientTimeProbe;

  if(m_MeasureExecutionTimes)
    {
    std::cout << "Starting ConjugateGradient" << std::endl;
    ConjugateGradientTimeProbe.Start();
    }

  m_ConjugateGradientFilter->Update();

  if(m_MeasureExecutionTimes)
    {
    ConjugateGradientTimeProbe.Stop();
    std::cout << "ConjugateGradient took " << ConjugateGradientTimeProbe.GetTotal() << ' ' << ConjugateGradientTimeProbe.GetUnit() << std::endl;
    }

  this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
}

}// end namespace


#endif
