#include "itkRayCastInterpolatorForwardProjectionImageFilter.h"

namespace itk
{

template<class TInputImage, class TOutputImage>
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::SARTConeBeamReconstructionFilter():
  m_NumberOfIterations(3),
  m_Lambda(0.3)
{
  typedef RayCastInterpolatorForwardProjectionImageFilter<TInputImage, TOutputImage> DefaultForwardType;
  this->SetNumberOfRequiredInputs(2);

  // Create each filter of the composite filter
  m_ExtractFilter = ExtractFilterType::New();
  m_ZeroMultiplyFilter = MultiplyFilterType::New();
  m_ForwardProjectionFilter = DefaultForwardType::New();
  m_SubtractFilter = SubtractFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_BackProjectionFilter = BackProjectionFilterType::New();

  //Permanent internal connections
  m_ZeroMultiplyFilter->SetInput( m_ExtractFilter->GetOutput() );
  m_ForwardProjectionFilter->SetInput( 0, m_ZeroMultiplyFilter->GetOutput() );
  m_SubtractFilter->SetInput(0, m_ExtractFilter->GetOutput() );
  m_SubtractFilter->SetInput(1, m_ForwardProjectionFilter->GetOutput() );
  m_MultiplyFilter->SetInput( m_SubtractFilter->GetOutput() );
  m_BackProjectionFilter->SetInput(1, m_MultiplyFilter->GetOutput() );

  // Default parameters
#if ITK_VERSION_MAJOR >= 4
  m_ExtractFilter->SetDirectionCollapseToSubmatrix();
#endif
  m_ZeroMultiplyFilter->SetConstant( 0. );
  m_BackProjectionFilter->InPlaceOn();
  m_BackProjectionFilter->SetTranspose(true);
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
  m_BackProjectionFilter->SetTranspose(false);
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

  if(this->GetGeometry().GetPointer() == NULL)
    itkGenericExceptionMacro(<< "The geometry of the reconstruction has not been set");

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
ThreeDCircularProjectionGeometry::Pointer
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::GetGeometry()
{
  return this->m_ForwardProjectionFilter->GetGeometry();
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg)
{
  itkDebugMacro("setting GeometryPointer to " << _arg);
  if (this->GetGeometry() != _arg)
    {
    m_ForwardProjectionFilter->SetGeometry(_arg);
    m_BackProjectionFilter->SetGeometry(_arg.GetPointer());
    this->Modified();
    }
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::PrintTiming(std::ostream& os) const
{
}

} // end namespace itk
