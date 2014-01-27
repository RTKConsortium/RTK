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
  m_DisplayExecutionTimes = false;       // Default behaviour : the execution
                                         // time of each filter is not displayed
                                         // on std::cout

  // Create each filter of the composite filter
  m_ExtractFilter = ExtractFilterType::New();
  m_ZeroMultiplyFilter = MultiplyFilterType::New();
  m_ForwardProjectionFilter = JosephForwardProjectionImageFilter<TInputImage, TOutputImage>::New();
  m_SubtractFilter = SubtractFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  //SetBackProjectionFilter(rtk::BackProjectionImageFilter<TInputImage,
  // TInputImage>::New());

  // Create the filters required for correct weighting of the difference
  // projection
  m_ExtractFilterRayBox = ExtractFilterType::New();
  m_RayBoxFilter = RayBoxIntersectionFilterType::New();
  m_DivideFilter = DivideFilterType::New();
  m_ZeroMultiplyFilterRayBox = MultiplyFilterType::New();

  //Permanent internal connections

#if ITK_VERSION_MAJOR >= 4
  m_ZeroMultiplyFilter->SetInput1( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
  m_ZeroMultiplyFilter->SetInput2( m_ExtractFilter->GetOutput() );
  m_ZeroMultiplyFilterRayBox->SetInput1( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
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

  //  m_RayBoxFilter->SetInput(m_ConstantImageSource->GetOutput());
  m_RayBoxFilter->SetInput(m_ZeroMultiplyFilterRayBox->GetOutput());
  m_ExtractFilterRayBox->SetInput(m_RayBoxFilter->GetOutput());
  m_DivideFilter->SetInput1(m_MultiplyFilter->GetOutput());
  m_DivideFilter->SetInput2(m_ExtractFilterRayBox->GetOutput());

  // Default parameters
#if ITK_VERSION_MAJOR >= 4
  m_ExtractFilter->SetDirectionCollapseToSubmatrix();
  m_ExtractFilterRayBox->SetDirectionCollapseToSubmatrix();
#else
  m_ZeroMultiplyFilter->SetConstant( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
  m_ZeroMultiplyFilterRayBox->SetConstant( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
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

  //SR: is this useful? CM : I think it isn't : GenerateInputRequestedRegion is
  // run after
  //GenerateOutputInformation. All the connections with inputs and
  // data-dependent
  //information should be set there, and do not have to be repeated here
  //
  //  m_BackProjectionFilter->SetInput ( 0, this->GetInput(0) );
  //  m_ForwardProjectionFilter->SetInput ( 1, this->GetInput(0) );
  //  m_ExtractFilter->SetInput( this->GetInput(1) );

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
  m_ExtractFilterRayBox->SetExtractionRegion(projRegion);

  // Run composite filter update
  m_BackProjectionFilter->SetInput ( 0, this->GetInput(0) );
  m_ForwardProjectionFilter->SetInput ( 1, this->GetInput(0) );
  m_ExtractFilter->SetInput( this->GetInput(1) );
#if ITK_VERSION_MAJOR >= 4
  m_ZeroMultiplyFilterRayBox->SetInput2( this->GetInput(1));
#else
  m_ZeroMultiplyFilterRayBox->SetInput( this->GetInput(1) );
#endif

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

  // Configure the extract filter to ask for the whole projection set
  m_ExtractFilterRayBox->SetExtractionRegion(this->GetInput(1)->GetLargestPossibleRegion());

  m_BackProjectionFilter->UpdateOutputInformation();

  // Update output information
  this->GetOutput()->SetOrigin( m_BackProjectionFilter->GetOutput()->GetOrigin() );
  this->GetOutput()->SetSpacing( m_BackProjectionFilter->GetOutput()->GetSpacing() );
  this->GetOutput()->SetDirection( m_BackProjectionFilter->GetOutput()->GetDirection() );
  this->GetOutput()->SetLargestPossibleRegion( m_BackProjectionFilter->GetOutput()->GetLargestPossibleRegion() );

  std::cout << "Beacon 1" << std::endl;
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

  DD("Starting iterations");

  // For each iteration, go over each projection
  for(unsigned int iter = 0; iter < m_NumberOfIterations; iter++)
    {
    for(unsigned int i = 0; i < nProj; i++)
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
      m_ExtractFilterRayBox->SetExtractionRegion(subsetRegion);

      // This is required to reset the full pipeline
      m_BackProjectionFilter->GetOutput()->UpdateOutputInformation();
      m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();

      //Initialize a time probe
      itk::TimeProbe                    timeProbe;
      itk::RealTimeClock::TimeStampType previousTotal;

      if (m_DisplayExecutionTimes)
        {
        timeProbe.Start();
        }

      m_ExtractFilter->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of extract filter took " << timeProbe.GetTotal() << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }

      m_ZeroMultiplyFilter->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of zero multiply filter took " << timeProbe.GetTotal() << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }

      m_ForwardProjectionFilter->UpdateLargestPossibleRegion();
      //            m_ForwardProjectionFilter->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of forward projection filter took " << timeProbe.GetTotal() - previousTotal << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }

      m_SubtractFilter->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of subtract filter took " << timeProbe.GetTotal() - previousTotal << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }

      m_MultiplyFilter->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of multiply by lambda filter took " << timeProbe.GetTotal() - previousTotal << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }

      m_ZeroMultiplyFilterRayBox->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of zero multiply filter for ray box took " << timeProbe.GetTotal() - previousTotal << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }

      m_RayBoxFilter->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of ray box intersection filter took " << timeProbe.GetTotal() - previousTotal << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }

      m_DivideFilter->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of divide filter took " << timeProbe.GetTotal() - previousTotal << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }

      m_BackProjectionFilter->Update();
      if (m_DisplayExecutionTimes)
        {
        timeProbe.Stop();
        std::cout << "Execution of back projection filter took " << timeProbe.GetTotal() - previousTotal << " " <<timeProbe.GetUnit() << std::endl;
        previousTotal = timeProbe.GetTotal();
        timeProbe.Start();
        }
      }
    }
  this->GraftOutput( m_BackProjectionFilter->GetOutput() );
}

template<class TInputImage, class TOutputImage>
void
SARTConeBeamReconstructionFilter<TInputImage, TOutputImage>
::PrintTiming(std::ostream & os) const
{}
} // end namespace rtk

#endif // __rtkSARTConeBeamReconstructionFilter_txx
