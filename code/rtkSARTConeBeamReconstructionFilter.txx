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

#include <algorithm>
#include "itkTimeProbe.h"

namespace rtk
{
template<class TInputImage, class TOutputImage>
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::SARTConeBeamReconstructionFilter() :
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

  // Create the filters required for correct weighting of the difference
  // projection
  m_ExtractFilterRayBox = ExtractFilterType::New();
  m_RayBoxFilter = RayBoxIntersectionFilterType::New();
  m_DivideFilter = DivideFilterType::New();
  m_ConstantImageSource = ConstantImageSourceType::New();

  // Create the filter that enforces positivity
  m_ThresholdFilter = ThresholdFilterType::New();
  
  //Permanent internal connections
#if ITK_VERSION_MAJOR >= 4
  m_ZeroMultiplyFilter->SetInput1( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
  m_ZeroMultiplyFilter->SetInput2( m_ExtractFilter->GetOutput() );
#else
  m_ZeroMultiplyFilter->SetInput( m_ExtractFilter->GetOutput() );
#endif
  m_ForwardProjectionFilter->SetInput( 0, m_ZeroMultiplyFilter->GetOutput() );
  m_SubtractFilter->SetInput(0, m_ExtractFilter->GetOutput() );
  m_SubtractFilter->SetInput(1, m_ForwardProjectionFilter->GetOutput() );
#if ITK_VERSION_MAJOR >= 4
  m_MultiplyFilter->SetInput1( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
  m_MultiplyFilter->SetInput2( m_SubtractFilter->GetOutput() );
#else
  m_MultiplyFilter->SetInput( m_SubtractFilter->GetOutput() );
#endif


  m_ExtractFilterRayBox->SetInput(m_ConstantImageSource->GetOutput());
  m_RayBoxFilter->SetInput(m_ExtractFilterRayBox->GetOutput());
  m_DivideFilter->SetInput1(m_MultiplyFilter->GetOutput());
  m_DivideFilter->SetInput2(m_RayBoxFilter->GetOutput());

  // Default parameters
#if ITK_VERSION_MAJOR >= 4
  m_ExtractFilter->SetDirectionCollapseToSubmatrix();
  m_ExtractFilterRayBox->SetDirectionCollapseToSubmatrix();
#else
  m_ZeroMultiplyFilter->SetConstant( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
#endif
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::SetBackProjectionFilter(const BackProjectionFilterPointer _arg)
{
  itkDebugMacro("setting BackProjectionFilter to " << _arg);
  if (this->m_BackProjectionFilter != _arg)
    {
    this->m_BackProjectionFilter = _arg;
    m_BackProjectionFilter->SetInput(1, m_DivideFilter->GetOutput() );
    m_BackProjectionFilter->SetTranspose(false);
    m_ThresholdFilter->SetInput(m_BackProjectionFilter->GetOutput() );
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

  m_ThresholdFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_ThresholdFilter->GetOutput()->PropagateRequestedRegion();

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
  m_ExtractFilterRayBox->SetExtractionRegion(projRegion);

  // Run composite filter update
  m_BackProjectionFilter->SetInput ( 0, this->GetInput(0) );
  m_ForwardProjectionFilter->SetInput ( 1, this->GetInput(0) );
  m_ExtractFilter->SetInput( this->GetInput(1) );

  m_ConstantImageSource->SetInformationFromImage(const_cast<TInputImage *>(this->GetInput(1)));
  m_ConstantImageSource->SetConstant(0);
  m_ConstantImageSource->UpdateOutputInformation();

  // Create the m_RayBoxFiltersectionImageFilter
  m_RayBoxFilter->SetGeometry(this->GetGeometry().GetPointer());
  itk::Vector<double, 3> Corner1, Corner2;

  Corner1[0] = this->GetInput(0)->GetOrigin()[0];
  Corner1[1] = this->GetInput(0)->GetOrigin()[1];
  Corner1[2] = this->GetInput(0)->GetOrigin()[2];
  Corner2[0] = this->GetInput(0)->GetOrigin()[0] + this->GetInput(0)->GetLargestPossibleRegion().GetSize()[0] * this->GetInput(0)->GetSpacing()[0];
  Corner2[1] = this->GetInput(0)->GetOrigin()[1] + this->GetInput(0)->GetLargestPossibleRegion().GetSize()[1] * this->GetInput(0)->GetSpacing()[1];
  Corner2[2] = this->GetInput(0)->GetOrigin()[2] + this->GetInput(0)->GetLargestPossibleRegion().GetSize()[2] * this->GetInput(0)->GetSpacing()[2];

  m_RayBoxFilter->SetBoxMin(Corner1);
  m_RayBoxFilter->SetBoxMax(Corner2);
  
  if(m_EnforcePositivity)
    {
    m_ThresholdFilter->SetOutsideValue(0);
    m_ThresholdFilter->ThresholdBelow(0);
    }
  m_ThresholdFilter->UpdateOutputInformation();

  // Update output information
  this->GetOutput()->SetOrigin( m_ThresholdFilter->GetOutput()->GetOrigin() );
  this->GetOutput()->SetSpacing( m_ThresholdFilter->GetOutput()->GetSpacing() );
  this->GetOutput()->SetDirection( m_ThresholdFilter->GetOutput()->GetDirection() );
  this->GetOutput()->SetLargestPossibleRegion( m_ThresholdFilter->GetOutput()->GetLargestPossibleRegion() );

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
  m_BackProjectionFilter->SetGeometry(this->GetGeometry().GetPointer());

  // The backprojection works on one projection at a time
  typename ExtractFilterType::InputImageRegionType subsetRegion;
  subsetRegion = this->GetInput(1)->GetLargestPossibleRegion();
  unsigned int nProj = subsetRegion.GetSize(Dimension-1);
  subsetRegion.SetSize(Dimension-1, 1);

  // Fill and shuffle randomly the projection order.
  // Should be tunable with other solutions.
  std::vector< unsigned int > projOrder(nProj);

  for(unsigned int i = 0; i < nProj; i++)
    projOrder[i] = i;
  std::random_shuffle( projOrder.begin(), projOrder.end() );

#if ITK_VERSION_MAJOR >= 4
  m_MultiplyFilter->SetInput1( (const float)m_Lambda  );
#else
  m_MultiplyFilter->SetConstant( m_Lambda );
#endif
  
  // Create the zero projection stack used as input by RayBoxIntersectionFilter
  m_ConstantImageSource->Update();
      
  // For each iteration, go over each projection
  for(unsigned int iter = 0; iter < m_NumberOfIterations; iter++)
    {
    for(unsigned int i = 0; i < nProj; i++)
      {
      // After the first bp update, we need to use its output as input.
      if(iter+i)
        {
        //typename TInputImage::Pointer pimg = m_BackProjectionFilter->GetOutput();
        typename TInputImage::Pointer pimg = m_ThresholdFilter->GetOutput();
        pimg->DisconnectPipeline();
        m_BackProjectionFilter->SetInput( pimg );
        m_ForwardProjectionFilter->SetInput(1, pimg );
        }

      // Change projection subset
      subsetRegion.SetIndex( Dimension-1, projOrder[i] );
      m_ExtractFilter->SetExtractionRegion(subsetRegion);
      m_ExtractFilterRayBox->SetExtractionRegion(subsetRegion);

      // This is required to reset the full pipeline
      m_BackProjectionFilter->GetOutput()->UpdateOutputInformation();
      m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();

      m_ExtractProbe.Start();
      m_ExtractFilter->Update();
      m_ExtractProbe.Stop();

      m_ZeroMultiplyProbe.Start();
      m_ZeroMultiplyFilter->Update();
      m_ZeroMultiplyProbe.Stop();

      m_ForwardProjectionProbe.Start();
      m_ForwardProjectionFilter->UpdateLargestPossibleRegion();
      //            m_ForwardProjectionFilter->Update();
      m_ForwardProjectionProbe.Stop();

      m_SubtractProbe.Start();
      m_SubtractFilter->Update();
      m_SubtractProbe.Stop();

      m_MultiplyProbe.Start();
      m_MultiplyFilter->Update();
      m_MultiplyProbe.Stop();

      m_ExtractFilterRayBox->Update();

      m_RayBoxProbe.Start();
      m_RayBoxFilter->Update();
      m_RayBoxProbe.Stop();

      m_DivideProbe.Start();
      m_DivideFilter->Update();
      m_DivideProbe.Stop();

      m_BackProjectionProbe.Start();
      m_BackProjectionFilter->Update();
      m_BackProjectionProbe.Stop();

      m_ThresholdProbe.Start();
      m_ThresholdFilter->Update();
      m_ThresholdProbe.Stop();
      }
    }
//  this->GraftOutput( m_BackProjectionFilter->GetOutput() );
  this->GraftOutput( m_ThresholdFilter->GetOutput());
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::PrintTiming(std::ostream & os) const
{
  os << "SARTConeBeamReconstructionFilter timing:" << std::endl;
  os << "  Extraction of projection sub-stacks: " << m_ExtractProbe.GetTotal()
     << ' ' << m_ExtractProbe.GetUnit() << std::endl;
  os << "  Multiplication by zero: " << m_ZeroMultiplyProbe.GetTotal()
     << ' ' << m_ZeroMultiplyProbe.GetUnit() << std::endl;
  os << "  Forward projection: " << m_ForwardProjectionProbe.GetTotal()
     << ' ' << m_ForwardProjectionProbe.GetUnit() << std::endl;
  os << "  Subtraction: " << m_SubtractProbe.GetTotal()
     << ' ' << m_SubtractProbe.GetUnit() << std::endl;
  os << "  Multiplication by lambda: " << m_MultiplyProbe.GetTotal()
     << ' ' << m_MultiplyProbe.GetUnit() << std::endl;
  os << "  Ray box intersection: " << m_RayBoxProbe.GetTotal()
     << ' ' << m_RayBoxProbe.GetUnit() << std::endl;
  os << "  Division: " << m_DivideProbe.GetTotal()
     << ' ' << m_DivideProbe.GetUnit() << std::endl;
  os << "  Back projection: " << m_BackProjectionProbe.GetTotal()
     << ' ' << m_BackProjectionProbe.GetUnit() << std::endl;
  if (m_EnforcePositivity)
    {  
    os << "  Positivity enforcement: " << m_ThresholdProbe.GetTotal()
    << ' ' << m_ThresholdProbe.GetUnit() << std::endl;
    }
  else
    {
    os << "  Positivity was not enforced, but passing through the filter still took: " << m_ThresholdProbe.GetTotal()
    << ' ' << m_ThresholdProbe.GetUnit() << std::endl;
    }
}

} // end namespace rtk

#endif // __rtkSARTConeBeamReconstructionFilter_txx
