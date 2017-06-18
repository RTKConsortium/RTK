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

#ifndef rtkReconstructionConjugateGradientOperator_hxx
#define rtkReconstructionConjugateGradientOperator_hxx

#include "rtkReconstructionConjugateGradientOperator.h"

namespace rtk
{

template< typename TOutputImage>
ReconstructionConjugateGradientOperator<TOutputImage>
::ReconstructionConjugateGradientOperator():
  m_Geometry(ITK_NULLPTR),
  m_Regularized(false),
  m_Gamma(0)
{
  this->SetNumberOfRequiredInputs(3);

  // Create filters
#ifdef RTK_USE_CUDA
  m_ConstantProjectionsSource = rtk::CudaConstantVolumeSource::New();
  m_ConstantVolumeSource = rtk::CudaConstantVolumeSource::New();
  m_LaplacianFilter = rtk::CudaLaplacianImageFilter::New();
#else
  m_ConstantProjectionsSource = ConstantSourceType::New();
  m_ConstantVolumeSource = ConstantSourceType::New();
  m_LaplacianFilter = LaplacianFilterType::New();
#endif
  m_MultiplyProjectionsFilter = MultiplyFilterType::New();
  m_MultiplyOutputVolumeFilter = MultiplyFilterType::New();
  m_MultiplyInputVolumeFilter = MultiplyFilterType::New();
  m_AddFilter = AddFilterType::New();

  m_MultiplyLaplacianFilter = MultiplyFilterType::New();

  // Set permanent parameters
  m_ConstantProjectionsSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_ConstantVolumeSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());

  // Set memory management options
  m_ConstantProjectionsSource->ReleaseDataFlagOn();
  m_ConstantVolumeSource->ReleaseDataFlagOn();
  m_LaplacianFilter->ReleaseDataFlagOn();
  m_MultiplyLaplacianFilter->ReleaseDataFlagOn();
}

template< typename TOutputImage>
void
ReconstructionConjugateGradientOperator<TOutputImage>::
SetSupportMask(const TOutputImage *SupportMask)
{
  this->SetInput("SupportMask", const_cast<TOutputImage*>(SupportMask));
}

template< typename TOutputImage>
typename TOutputImage::ConstPointer
ReconstructionConjugateGradientOperator<TOutputImage>::
GetSupportMask()
{
  return static_cast< const TOutputImage * >
          ( this->itk::ProcessObject::GetInput("SupportMask") );
}

template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
::SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg)
{
  m_BackProjectionFilter = _arg;
}

template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
::SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg)
{
  m_ForwardProjectionFilter = _arg;
}

template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
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

  // Input 2 is the weights map on projections, if any
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

template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections, and connections with
  // forward and back projection filters, which are set
  // at runtime
  m_ConstantVolumeSource->SetInformationFromImage(this->GetInput(0));
  m_ConstantProjectionsSource->SetInformationFromImage(this->GetInput(1));

  m_FloatingInputPointer = const_cast<TOutputImage *>(this->GetInput(0));

  // Set the first multiply filter to use the Support Mask, if any
  if (this->GetSupportMask().IsNotNull())
    {
    m_MultiplyInputVolumeFilter->SetInput1( m_FloatingInputPointer );
    m_MultiplyInputVolumeFilter->SetInput2( this->GetSupportMask() );
    m_FloatingInputPointer = m_MultiplyInputVolumeFilter->GetOutput();
    }

  // Set the forward projection filter's inputs
  m_ForwardProjectionFilter->SetInput(0, m_ConstantProjectionsSource->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, m_FloatingInputPointer);

  // Set the multiply filter's inputs for the projection weights (for WLS minimization)
  m_MultiplyProjectionsFilter->SetInput1(m_ForwardProjectionFilter->GetOutput());
  m_MultiplyProjectionsFilter->SetInput2(this->GetInput(2));

  // Set the back projection filter's inputs
  m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_MultiplyProjectionsFilter->GetOutput());
  m_FloatingOutputPointer= m_BackProjectionFilter->GetOutput();

  // Set the filters to compute the regularization, if any
  if (m_Regularized)
    {
    m_LaplacianFilter->SetInput(m_FloatingInputPointer);
    m_MultiplyLaplacianFilter->SetInput1(m_LaplacianFilter->GetOutput());
    // Set "-1.0*gamma" because we need to perform "-1.0*Laplacian"
    // for correctly applying quadratic regularization || grad f ||_2^2
    m_MultiplyLaplacianFilter->SetConstant2(-1.0*m_Gamma);

    m_AddFilter->SetInput1( m_BackProjectionFilter->GetOutput());
    m_AddFilter->SetInput2( m_MultiplyLaplacianFilter->GetOutput());

    m_FloatingOutputPointer= m_AddFilter->GetOutput();
    }

  // Set the second multiply filter to use the Support Mask, if any
  if (this->GetSupportMask().IsNotNull())
    {
    m_MultiplyOutputVolumeFilter->SetInput1( m_FloatingOutputPointer);
    m_MultiplyOutputVolumeFilter->SetInput2( this->GetSupportMask() );
    m_FloatingOutputPointer= m_MultiplyOutputVolumeFilter->GetOutput();
    }

  // Set geometry
  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());

  // Set memory management parameters for forward
  // and back projection filters
  m_ForwardProjectionFilter->SetInPlace(true);
  m_ForwardProjectionFilter->ReleaseDataFlagOn();
  m_BackProjectionFilter->SetInPlace(true);
  m_BackProjectionFilter->SetReleaseDataFlag(this->GetSupportMask().IsNotNull() || m_Regularized);

  // Update output information on the last filter of the pipeline
  m_FloatingOutputPointer->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( m_FloatingOutputPointer);
}

template< typename TOutputImage >
void ReconstructionConjugateGradientOperator<TOutputImage>::GenerateData()
{
  // Execute Pipeline
  m_FloatingOutputPointer->Update();
  this->GraftOutput( m_FloatingOutputPointer );
}

}// end namespace


#endif
