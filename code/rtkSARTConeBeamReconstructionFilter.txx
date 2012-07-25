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

#ifndef __rtkSARTConeBeamReconstructionFilter_txx
#define __rtkSARTConeBeamReconstructionFilter_txx

#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"

namespace rtk
{

template<class TInputImage, class TOutputImage>
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::SARTConeBeamReconstructionFilter():
  m_NumberOfIterations(3),
  m_Lambda(0.3)
{
  this->SetNumberOfRequiredInputs(2);

  // Create each filter of the composite filter
  m_ExtractFilter = ExtractFilterType::New();
  m_ZeroMultiplyFilter = MultiplyFilterType::New();
  m_ForwardProjectionFilter = JosephForwardProjectionImageFilter<TInputImage, TOutputImage>::New();
  m_SubtractFilter = SubtractFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  SetBackProjectionFilter(rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::New());

  //Permanent internal connections
  m_ZeroMultiplyFilter->SetInput( m_ExtractFilter->GetOutput() );
  m_ForwardProjectionFilter->SetInput( 0, m_ZeroMultiplyFilter->GetOutput() );
  m_SubtractFilter->SetInput(0, m_ExtractFilter->GetOutput() );
  m_SubtractFilter->SetInput(1, m_ForwardProjectionFilter->GetOutput() );
  m_MultiplyFilter->SetInput( m_SubtractFilter->GetOutput() );

  // Default parameters
#if ITK_VERSION_MAJOR >= 4
  m_ExtractFilter->SetDirectionCollapseToSubmatrix();
#endif
  m_ZeroMultiplyFilter->SetConstant( 0. );
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::SetBackProjectionFilter (const BackProjectionFilterPointer _arg)
{
  itkDebugMacro("setting BackProjectionFilter to " << _arg);
  if (this->m_BackProjectionFilter != _arg)
    {
    this->m_BackProjectionFilter = _arg;
    m_BackProjectionFilter->SetInput(1, m_MultiplyFilter->GetOutput() );
    m_BackProjectionFilter->SetTranspose(false);
    this->Modified();
    }
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer inputPtr =
    const_cast< TInputImage * >( this->GetInput() );
  if ( !inputPtr )
    return;

  //SR: is this useful?
  m_BackProjectionFilter->SetInput ( 0, this->GetInput(0) );
  m_ForwardProjectionFilter->SetInput ( 1, this->GetInput(0) );
  m_ExtractFilter->SetInput( this->GetInput(1) );
  m_BackProjectionFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  const unsigned int Dimension = this->InputImageDimension;

  // We only set the first sub-stack at that point, the rest will be
  // requested in the GenerateData function
  typename ExtractFilterType::InputImageRegionType projRegion;
  projRegion = this->GetInput(1)->GetLargestPossibleRegion();
  projRegion.SetSize(Dimension-1, 1);
  m_ExtractFilter->SetExtractionRegion(projRegion);

  // Run composite filter update
  m_BackProjectionFilter->SetInput ( 0, this->GetInput(0) );
  m_ForwardProjectionFilter->SetInput ( 1, this->GetInput(0) );
  m_ExtractFilter->SetInput( this->GetInput(1) );
  m_BackProjectionFilter->UpdateOutputInformation();

  // Update output information
  this->GetOutput()->SetOrigin( m_BackProjectionFilter->GetOutput()->GetOrigin() );
  this->GetOutput()->SetSpacing( m_BackProjectionFilter->GetOutput()->GetSpacing() );
  this->GetOutput()->SetDirection( m_BackProjectionFilter->GetOutput()->GetDirection() );
  this->GetOutput()->SetLargestPossibleRegion( m_BackProjectionFilter->GetOutput()->GetLargestPossibleRegion() );
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::GenerateData()
{
  const unsigned int Dimension = this->InputImageDimension;

  // Check and set geometry
  if(this->GetGeometry().GetPointer() == NULL)
    itkGenericExceptionMacro(<< "The geometry of the reconstruction has not been set");
  m_ForwardProjectionFilter->SetGeometry(this->GetGeometry().GetPointer());
  m_BackProjectionFilter   ->SetGeometry(this->GetGeometry().GetPointer());

  // The backprojection works on one projection at a time
  typename ExtractFilterType::InputImageRegionType subsetRegion;
  subsetRegion = this->GetInput(1)->GetLargestPossibleRegion();
  unsigned int nProj = subsetRegion.GetSize(Dimension-1);
  subsetRegion.SetSize(Dimension-1, 1);

  // Fill and shuffle randomly the projection order.
  // Should be tunable with other solutions.
  std::vector< unsigned int > projOrder(nProj);
  for(unsigned int i=0; i<nProj; i++)
    projOrder[i] = i;
  std::random_shuffle( projOrder.begin(), projOrder.end() );

  // Set convergence factor. Approximate ray length through box with the
  // largest possible length through volume (volume diagonal).
  typename TOutputImage::SpacingType sizeInMM;
  for(unsigned int i=0; i<Dimension; i++)
    sizeInMM[i] = this->GetInput(0)->GetLargestPossibleRegion().GetSize()[i] *
                  this->GetInput(0)->GetSpacing()[i];
  m_MultiplyFilter->SetConstant( m_Lambda / sizeInMM.GetNorm() );

  // For each iteration, go over each projection
  for(unsigned int iter=0; iter<m_NumberOfIterations; iter++)
    {
    for(unsigned int i=0; i<nProj; i++)
      {
      // After the first bp update, we need to use its output as input.
      if(iter+i)
        {
        typename TInputImage::Pointer pimg = m_BackProjectionFilter->GetOutput();
        pimg->DisconnectPipeline();
        m_BackProjectionFilter->SetInput( pimg );
        m_ForwardProjectionFilter->SetInput(1, pimg );
        }

      // Change projection subset
      subsetRegion.SetIndex( Dimension-1, projOrder[i] );
      m_ExtractFilter->SetExtractionRegion(subsetRegion);
      // This is required to reset the full pipeline
      m_BackProjectionFilter->GetOutput()->UpdateOutputInformation();
      m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();

      m_ExtractFilter->Update();
      m_ZeroMultiplyFilter->Update();
      m_ForwardProjectionFilter->Update();
      m_SubtractFilter->Update();
      m_MultiplyFilter->Update();
      m_BackProjectionFilter->Update();

//if(i%32==0)
//  {
//  typedef typename itk::ImageFileWriter<TOutputImage> WriterType;
//  typename WriterType::Pointer writer = WriterType::New();
//  writer->SetFileName("sart.mha");
//  writer->SetInput( m_BackProjectionFilter->GetOutput() );
//  writer->Update();
//  }

      }
    }
  GraftOutput( m_BackProjectionFilter->GetOutput() );
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::PrintTiming(std::ostream& os) const
{
}

} // end namespace rtk

#endif // __rtkSARTConeBeamReconstructionFilter_txx
