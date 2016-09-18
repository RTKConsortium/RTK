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

#ifndef rtkConjugateGradientConeBeamReconstructionFilter_hxx
#define rtkConjugateGradientConeBeamReconstructionFilter_hxx

#include "rtkConjugateGradientConeBeamReconstructionFilter.h"

namespace rtk
{

template< typename TOutputImage>
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>::ConjugateGradientConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(3);

  // Set the default values of member parameters
  m_NumberOfIterations=3;
  m_MeasureExecutionTimes=false;
  m_IterationCosts=false;
  m_Preconditioned=false;
  m_Gamma = 0;
  m_Regularized = false;
  m_CudaConjugateGradient = true;

  // Create the filters
#ifdef RTK_USE_CUDA
  m_DisplacedDetectorFilter = rtk::CudaDisplacedDetectorImageFilter::New();
  m_ConstantVolumeSource     = rtk::CudaConstantVolumeSource::New();
#else
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
  m_ConstantVolumeSource     = ConstantImageSourceType::New();
#endif
  m_CGOperator = CGOperatorFilterType::New();

  m_DivideFilter = DivideFilterType::New();
  m_ConstantProjectionsSource = ConstantImageSourceType::New();

  m_MultiplyVolumeFilter = MultiplyFilterType::New();
  m_MultiplyProjectionsFilter = MultiplyFilterType::New();
  m_MultiplyOutputFilter = MultiplyFilterType::New();
  m_MultiplySupportMaskFilter = MultiplyFilterType::New();
  m_MultiplySupportMaskFilterForOutput = MultiplyFilterType::New();

  // Set permanent parameters
  m_ConstantVolumeSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_ConstantProjectionsSource->SetConstant(1.0);
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
}


template< typename TOutputImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>::
SetSupportMask(const TOutputImage *SupportMask)
{
  this->SetInput("SupportMask", const_cast<TOutputImage*>(SupportMask));
}

template< typename TOutputImage>
typename TOutputImage::ConstPointer
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>::
GetSupportMask()
{
  return static_cast< const TOutputImage * >
          ( this->itk::ProcessObject::GetInput("SupportMask") );
}

template< typename TOutputImage>
const std::vector<double> &ConjugateGradientConeBeamReconstructionFilter<TOutputImage>
::GetResidualCosts()
{
  return m_ConjugateGradientFilter->GetResidualCosts();
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
    m_BackProjectionFilterForPreconditioning = this->InstantiateBackProjectionFilter( _arg );
    m_BackProjectionFilterForNormalization = this->InstantiateBackProjectionFilter( _arg );
    m_CGOperator->SetBackProjectionFilter( m_BackProjectionFilter);
    }
}

template< typename TOutputImage >
const TOutputImage *
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>
::ApplySupportMask (const TOutputImage *_arg)
{
  if (this->GetSupportMask().IsNotNull())
  {
    m_MultiplySupportMaskFilter->SetInput(0,_arg);
    m_MultiplySupportMaskFilter->Update();
    return m_MultiplySupportMaskFilter->GetOutput();
  }
  else
  {
    return _arg;
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

  // Input 2 is the weights map on projections, either user-defined or filled with ones (default)
  typename Superclass::InputImagePointer  inputPtr2 =
          const_cast< TOutputImage * >( this->GetInput(2) );
  if ( !inputPtr2 )
      return;
  inputPtr2->SetRequestedRegion( inputPtr2->GetLargestPossibleRegion() );

  // Input "SupportMask" is the support constraint mask on volume, if any
  if (this->GetSupportMask().IsNotNull())
    {
    typename Superclass::InputImagePointer inputSupportMaskPtr =
            const_cast< TOutputImage * >( this->GetSupportMask().GetPointer() );
    if ( !inputSupportMaskPtr )
        return;
    inputSupportMaskPtr->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );
    }
}

template< typename TOutputImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage>
::GenerateOutputInformation()
{
  // Choose between cuda or non-cuda conjugate gradient filter
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
#ifdef RTK_USE_CUDA
  if (m_CudaConjugateGradient)
    m_ConjugateGradientFilter = rtk::CudaConjugateGradientImageFilter_3f::New();
#endif
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());
  m_ConjugateGradientFilter->SetIterationCosts(m_IterationCosts);
  
  // Set runtime connections
  m_ConstantVolumeSource->SetInformationFromImage(this->GetInput(0));
  m_CGOperator->SetInput(1, this->GetInput(1));
  m_CGOperator->SetSupportMask(this->GetSupportMask());
  m_ConjugateGradientFilter->SetX(this->GetInput(0));
  m_DisplacedDetectorFilter->SetInput(this->GetInput(2));

  // Multiply the projections by the weights map
  m_MultiplyProjectionsFilter->SetInput1(this->GetInput(1));  
  m_MultiplyProjectionsFilter->SetInput2(m_DisplacedDetectorFilter->GetOutput());
  m_CGOperator->SetInput(2, m_DisplacedDetectorFilter->GetOutput());
  m_BackProjectionFilterForB->SetInput(1, m_MultiplyProjectionsFilter->GetOutput());

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_BackProjectionFilterForB->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_ConjugateGradientFilter->SetB(m_BackProjectionFilterForB->GetOutput());

  // Multiply the projections by the weights map
  m_MultiplyProjectionsFilter->SetInput1(this->GetInput(1));  
  m_MultiplyProjectionsFilter->SetInput2(m_DisplacedDetectorFilter->GetOutput());
  m_CGOperator->SetInput(2, m_DisplacedDetectorFilter->GetOutput());
  m_BackProjectionFilterForB->SetInput(1, m_MultiplyProjectionsFilter->GetOutput());

  if (this->GetSupportMask().IsNotNull())
    {
    m_MultiplySupportMaskFilter->SetInput(0,m_BackProjectionFilterForB->GetOutput());
    m_MultiplySupportMaskFilter->SetInput(1,this->GetSupportMask());
    m_ConjugateGradientFilter->SetB(m_MultiplySupportMaskFilter->GetOutput());   
    }

  if (m_Preconditioned)
    {
    // Set the projections source
    m_ConstantProjectionsSource->SetInformationFromImage(this->GetInput(1));

    // Build the part of the pipeline that generates the preconditioning weights
    m_BackProjectionFilterForNormalization->SetInput(0, m_ConstantVolumeSource->GetOutput());
    m_BackProjectionFilterForNormalization->SetInput(1, m_ConstantProjectionsSource->GetOutput());
    m_BackProjectionFilterForPreconditioning->SetInput(0, m_ConstantVolumeSource->GetOutput());
    m_BackProjectionFilterForPreconditioning->SetInput(1, m_DisplacedDetectorFilter->GetOutput());
  
    m_DivideFilter->SetInput1(m_BackProjectionFilterForNormalization->GetOutput());
    m_DivideFilter->SetInput2(m_BackProjectionFilterForPreconditioning->GetOutput());

    // Multiply the volume by preconditioning weights, and pass them to the conjugate gradient operator
    m_MultiplyVolumeFilter->SetInput1(m_BackProjectionFilterForB->GetOutput());
    m_MultiplyVolumeFilter->SetInput2(m_DivideFilter->GetOutput());
    m_CGOperator->SetInput(3, m_DivideFilter->GetOutput());
    m_ConjugateGradientFilter->SetB(m_MultiplyVolumeFilter->GetOutput());

    if (this->GetSupportMask().IsNotNull())
    {
    m_MultiplySupportMaskFilter->SetInput(0,m_BackProjectionFilterForB->GetOutput());
    m_MultiplySupportMaskFilter->SetInput(1,this->GetSupportMask());
    m_MultiplyVolumeFilter->SetInput1(m_MultiplySupportMaskFilter->GetOutput());
    }

    // Divide the output by the preconditioning weights
    m_MultiplyOutputFilter->SetInput1(m_ConjugateGradientFilter->GetOutput());
    m_MultiplyOutputFilter->SetInput2(m_DivideFilter->GetOutput());
    }

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->m_Geometry);
  m_BackProjectionFilterForB->SetGeometry(this->m_Geometry.GetPointer());
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);
  m_BackProjectionFilterForNormalization->SetGeometry(this->m_Geometry.GetPointer());
  m_BackProjectionFilterForPreconditioning->SetGeometry(this->m_Geometry.GetPointer());
  
  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);
  m_CGOperator->SetPreconditioned(m_Preconditioned);
  m_CGOperator->SetRegularized(m_Regularized);
  m_CGOperator->SetGamma(m_Gamma);

  // Set memory management parameters
  m_MultiplyProjectionsFilter->ReleaseDataFlagOn();

  if (m_Preconditioned)
    {
    m_ConstantProjectionsSource->ReleaseDataFlagOn();
    m_BackProjectionFilterForPreconditioning->ReleaseDataFlagOn();
    m_BackProjectionFilterForNormalization->ReleaseDataFlagOn();
    m_MultiplyVolumeFilter->ReleaseDataFlagOn();
    m_MultiplyOutputFilter->ReleaseDataFlagOn();
    }
  m_BackProjectionFilterForB->ReleaseDataFlagOn();

  if (this->GetSupportMask().IsNotNull())
    {
    m_MultiplySupportMaskFilterForOutput->SetInput(1,this->GetSupportMask());
    if (m_Preconditioned)
      {
      m_MultiplySupportMaskFilterForOutput->SetInput(0,m_MultiplyOutputFilter->GetOutput());
      }
    else
      {
      m_MultiplySupportMaskFilterForOutput->SetInput(0,m_ConjugateGradientFilter->GetOutput());
      }
    }

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
  typename StatisticsImageFilterType::Pointer StatisticsImageFilterForC = StatisticsImageFilterType::New();
  typename MultiplyFilterType::Pointer MultiplyFilterForC = MultiplyFilterType::New();

  if (m_IterationCosts)
  {
      MultiplyFilterForC->SetInput(0,this->GetInput(1));
      MultiplyFilterForC->SetInput(1,this->GetInput(2));
      MultiplyFilterForC->Update();
      MultiplyFilterForC->SetInput(1,MultiplyFilterForC->GetOutput());
      MultiplyFilterForC->Update();
      StatisticsImageFilterForC->SetInput(MultiplyFilterForC->GetOutput());
      StatisticsImageFilterForC->Update();
      m_ConjugateGradientFilter->SetC(0.5*StatisticsImageFilterForC->GetSum());
  }

  if(m_MeasureExecutionTimes)
    {
    std::cout << "Starting ConjugateGradient" << std::endl;
    ConjugateGradientTimeProbe.Start();
    }

  if (m_Preconditioned)
    m_DivideFilter->Update();

  m_ConjugateGradientFilter->Update();

  if (m_Preconditioned)
    m_MultiplyOutputFilter->Update();

  if (this->GetSupportMask())
    {
    m_MultiplySupportMaskFilter->Update();
    m_MultiplySupportMaskFilterForOutput->Update();
    }

  if(m_MeasureExecutionTimes)
    {
    ConjugateGradientTimeProbe.Stop();
    std::cout << "ConjugateGradient took " << ConjugateGradientTimeProbe.GetTotal() << ' ' << ConjugateGradientTimeProbe.GetUnit() << std::endl;
    }

  if (this->GetSupportMask())
    {
    this->GraftOutput( m_MultiplySupportMaskFilterForOutput->GetOutput() );
    }
  else
    {
      if (m_Preconditioned)
      {
      this->GraftOutput( m_MultiplyOutputFilter->GetOutput() );
      }
      else
      {
      this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
      }
    }
}

}// end namespace


#endif
