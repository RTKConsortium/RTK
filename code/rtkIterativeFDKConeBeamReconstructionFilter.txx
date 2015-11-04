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

#ifndef __rtkIterativeFDKConeBeamReconstructionFilter_txx
#define __rtkIterativeFDKConeBeamReconstructionFilter_txx

#include "rtkIterativeFDKConeBeamReconstructionFilter.h"

#include <algorithm>
#include <itkTimeProbe.h>

namespace rtk
{
template<class TInputImage, class TOutputImage, class TFFTPrecision>
IterativeFDKConeBeamReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>
::IterativeFDKConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set default parameters
  m_EnforcePositivity = false;
  m_NumberOfIterations = 3;
  m_Lambda = 0.3;
  m_TruncationCorrection = 0.0;
  m_HannCutFrequency = 0.0;
  m_HannCutFrequencyY = 0.0;
  m_ProjectionSubsetSize = 16;

  // Create each filter of the composite filter
  typedef rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter<OutputImageType, OutputImageType> DisplacedDetectorForOffsetFieldOfViewType;
  m_DisplacedDetectorFilter = DisplacedDetectorForOffsetFieldOfViewType::New();
  m_ParkerFilter = ParkerFilterType::New();
  m_FDKFilter = FDKFilterType::New();
  m_ThresholdFilter = ThresholdFilterType::New();
  m_SubtractFilter = SubtractFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_ConstantProjectionStackSource = ConstantImageSourceType::New();

  // Filter parameters
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(true);
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
IterativeFDKConeBeamReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>
::SetForwardProjectionFilter (int _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter( _arg );
    }
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
IterativeFDKConeBeamReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer inputPtr =
    const_cast< TInputImage * >( this->GetInput() );

  if ( !inputPtr )
    return;

  m_FDKFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
  m_FDKFilter->GetOutput()->PropagateRequestedRegion();
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
IterativeFDKConeBeamReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>
::GenerateOutputInformation()
{
  // Set FDK parameters
  m_FDKFilter->GetRampFilter()->SetTruncationCorrection(m_TruncationCorrection);
  m_FDKFilter->GetRampFilter()->SetHannCutFrequency(m_HannCutFrequency);
  m_FDKFilter->GetRampFilter()->SetHannCutFrequencyY(m_HannCutFrequencyY);
  m_FDKFilter->SetProjectionSubsetSize(m_ProjectionSubsetSize);

  // Source
  m_ConstantProjectionStackSource->SetInformationFromImage(const_cast<TInputImage *>(this->GetInput(1)));
  m_ConstantProjectionStackSource->SetConstant(0);
  m_ConstantProjectionStackSource->UpdateOutputInformation();

  //Initial internal connections
  m_DisplacedDetectorFilter->SetInput( this->GetInput(1) );

  m_ParkerFilter->SetInput( m_DisplacedDetectorFilter->GetOutput() );

  m_FDKFilter->SetInput( 0, this->GetInput(0) );
  m_FDKFilter->SetInput( 1, m_ParkerFilter->GetOutput() );

  m_ForwardProjectionFilter->SetInput(0, m_ConstantProjectionStackSource->GetOutput() );
  m_ForwardProjectionFilter->SetInput(1, m_FDKFilter->GetOutput() );

  m_SubtractFilter->SetInput1( this->GetInput(1) );
  m_SubtractFilter->SetInput2( m_ForwardProjectionFilter->GetOutput() );

  m_MultiplyFilter->SetInput1( m_SubtractFilter->GetOutput() );
  m_MultiplyFilter->SetConstant2( m_Lambda );

  // Check and set geometry
  if(this->GetGeometry().GetPointer() == NULL)
    {
    itkGenericExceptionMacro(<< "The geometry of the reconstruction has not been set");
    }
  m_DisplacedDetectorFilter->SetGeometry(m_Geometry);
  m_ParkerFilter->SetGeometry(m_Geometry);
  m_FDKFilter->SetGeometry(m_Geometry);
  m_ForwardProjectionFilter->SetGeometry(m_Geometry);

  // Slightly modify the pipeline if positivity enforcement is requested
  if(m_EnforcePositivity)
    {
    m_ThresholdFilter->SetOutsideValue(0);
    m_ThresholdFilter->ThresholdBelow(0);
    m_ThresholdFilter->SetInput(m_FDKFilter->GetOutput() );
    m_ForwardProjectionFilter->SetInput(1, m_ThresholdFilter->GetOutput() );
    }

  // Update output information on the last filter of the pipeline
  m_MultiplyFilter->UpdateOutputInformation();

  // Copy the information from the filter that will actually return the output
  if(m_EnforcePositivity)
    this->GetOutput()->CopyInformation( m_ThresholdFilter->GetOutput() );
  else
    this->GetOutput()->CopyInformation( m_FDKFilter->GetOutput() );

  // Set memory management flags
  // TODO
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
IterativeFDKConeBeamReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>
::GenerateData()
{
  // Declare the image pointers used in the main loop
  typename TInputImage::Pointer p_projs;
  typename TInputImage::Pointer p_vol;

  // Run the first reconstruction
  if(m_EnforcePositivity)
    m_ThresholdFilter->Update();
  else
    m_FDKFilter->Update();

  // For each iteration over 1, go over each projection
  for(unsigned int iter = 1; iter < m_NumberOfIterations; iter++)
    {
    m_MultiplyFilter->Update();

    // Use previous iteration's result as input volume in next iteration
    if(m_EnforcePositivity)
      p_vol = m_ThresholdFilter->GetOutput();
    else
      p_vol = m_FDKFilter->GetOutput();
    p_vol->DisconnectPipeline();
    m_FDKFilter->SetInput( 0, p_vol );

    // Recreate broken link
    if(m_EnforcePositivity)
      m_ForwardProjectionFilter->SetInput(1, m_ThresholdFilter->GetOutput() );
    else
      m_ForwardProjectionFilter->SetInput(1, m_FDKFilter->GetOutput() );

    // Use correction projections as input projections in next iteration
    // No broken link to re-create
    p_projs = m_MultiplyFilter->GetOutput();
    p_projs->DisconnectPipeline();
    m_DisplacedDetectorFilter->SetInput( p_projs );

    // Run the next reconstruction
    if(m_EnforcePositivity)
      m_ThresholdFilter->Update();
    else
      m_FDKFilter->Update();
    }

  if (m_EnforcePositivity)
    this->GraftOutput( m_ThresholdFilter->GetOutput() );
  else
    this->GraftOutput( m_FDKFilter->GetOutput() );
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
IterativeFDKConeBeamReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>
::PrintTiming(std::ostream & os) const
{
//  os << "IterativeFDKConeBeamReconstructionFilter timing:" << std::endl;
//  os << "  Extraction of projection sub-stacks: " << m_ExtractProbe.GetTotal()
//     << ' ' << m_ExtractProbe.GetUnit() << std::endl;
//  os << "  Multiplication by zero: " << m_ZeroMultiplyProbe.GetTotal()
//     << ' ' << m_ZeroMultiplyProbe.GetUnit() << std::endl;
//  os << "  Forward projection: " << m_ForwardProjectionProbe.GetTotal()
//     << ' ' << m_ForwardProjectionProbe.GetUnit() << std::endl;
//  os << "  Subtraction: " << m_SubtractProbe.GetTotal()
//     << ' ' << m_SubtractProbe.GetUnit() << std::endl;
//  os << "  Multiplication by lambda: " << m_MultiplyProbe.GetTotal()
//     << ' ' << m_MultiplyProbe.GetUnit() << std::endl;
//  os << "  Ray box intersection: " << m_RayBoxProbe.GetTotal()
//     << ' ' << m_RayBoxProbe.GetUnit() << std::endl;
//  os << "  Division: " << m_DivideProbe.GetTotal()
//     << ' ' << m_DivideProbe.GetUnit() << std::endl;
//  os << "  Multiplication by the gating weights: " << m_GatingProbe.GetTotal()
//     << ' ' << m_GatingProbe.GetUnit() << std::endl;
//  os << "  Displaced detector: " << m_DisplacedDetectorProbe.GetTotal()
//     << ' ' << m_DisplacedDetectorProbe.GetUnit() << std::endl;
//  os << "  Back projection: " << m_BackProjectionProbe.GetTotal()
//     << ' ' << m_BackProjectionProbe.GetUnit() << std::endl;
//  os << "  Volume update: " << m_AddProbe.GetTotal()
//     << ' ' << m_AddProbe.GetUnit() << std::endl;
//  if (m_EnforcePositivity)
//    {
//    os << "  Positivity enforcement: " << m_ThresholdProbe.GetTotal()
//    << ' ' << m_ThresholdProbe.GetUnit() << std::endl;
//    }
}

} // end namespace rtk

#endif // __rtkIterativeFDKConeBeamReconstructionFilter_txx
