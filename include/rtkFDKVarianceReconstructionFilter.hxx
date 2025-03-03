/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkFDKVarianceReconstructionFilter_hxx
#define rtkFDKVarianceReconstructionFilter_hxx

#include "rtkFDKVarianceReconstructionFilter.h"

#include <itkProgressAccumulator.h>

namespace rtk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
FDKVarianceReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>::FDKVarianceReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Create each filter of the composite filter
  m_ExtractFilter = ExtractFilterType::New();
  m_WeightFilter1 = WeightFilterType::New();
  m_WeightFilter2 = WeightFilterType::New();
  m_VarianceRampFilter = VarianceRampFilterType::New();
  this->SetBackProjectionFilter(BackProjectionFilterType::New());

  // Permanent internal connections
  m_WeightFilter1->SetInput(m_ExtractFilter->GetOutput());
  m_WeightFilter2->SetInput(m_WeightFilter1->GetOutput());
  m_VarianceRampFilter->SetInput(m_WeightFilter2->GetOutput());

  // Default parameters
  m_ExtractFilter->SetDirectionCollapseToSubmatrix();
  m_WeightFilter1->InPlaceOn();
  m_WeightFilter2->InPlaceOn();

  // Default to one projection per subset when FFTW is not available
#if !defined(USE_FFTWD)
  if (typeid(TFFTPrecision).name() == typeid(double).name())
    m_ProjectionSubsetSize = 2;
#endif
#if !defined(USE_FFTWF)
  if (typeid(TFFTPrecision).name() == typeid(float).name())
    m_ProjectionSubsetSize = 2;
#endif
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FDKVarianceReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FDKVarianceReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer inputPtr = const_cast<TInputImage *>(this->GetInput());
  if (!inputPtr)
    return;

  // SR: is this useful?
  m_BackProjectionFilter->SetInput(0, this->GetInput(0));
  m_BackProjectionFilter->SetInPlace(this->GetInPlace());
  m_ExtractFilter->SetInput(this->GetInput(1));
  m_BackProjectionFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
  m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FDKVarianceReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>::GenerateOutputInformation()
{
  const unsigned int Dimension = this->InputImageDimension;

  m_WeightFilter1->SetGeometry(m_Geometry);
  m_WeightFilter2->SetGeometry(m_Geometry);
  m_BackProjectionFilter->SetGeometry(m_Geometry);

  // We only set the first sub-stack at that point, the rest will be
  // requested in the GenerateData function
  typename ExtractFilterType::InputImageRegionType projRegion;
  projRegion = this->GetInput(1)->GetLargestPossibleRegion();
  unsigned int firstStackSize = std::min(m_ProjectionSubsetSize, (unsigned int)projRegion.GetSize(Dimension - 1));
  projRegion.SetSize(Dimension - 1, firstStackSize);
  m_ExtractFilter->SetExtractionRegion(projRegion);

  // Run composite filter update
  m_BackProjectionFilter->SetInput(0, this->GetInput(0));
  m_BackProjectionFilter->SetInPlace(this->GetInPlace());
  m_ExtractFilter->SetInput(this->GetInput(1));
  m_BackProjectionFilter->UpdateOutputInformation();

  // Update output information
  this->GetOutput()->SetOrigin(m_BackProjectionFilter->GetOutput()->GetOrigin());
  this->GetOutput()->SetSpacing(m_BackProjectionFilter->GetOutput()->GetSpacing());
  this->GetOutput()->SetDirection(m_BackProjectionFilter->GetOutput()->GetDirection());
  this->GetOutput()->SetLargestPossibleRegion(m_BackProjectionFilter->GetOutput()->GetLargestPossibleRegion());
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FDKVarianceReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>::GenerateData()
{
  const unsigned int Dimension = this->InputImageDimension;

  // The backprojection works on a small stack of projections, not the full stack
  typename ExtractFilterType::InputImageRegionType subsetRegion;
  subsetRegion = this->GetInput(1)->GetLargestPossibleRegion();
  unsigned int nProj = subsetRegion.GetSize(Dimension - 1);

  // The progress accumulator tracks the progress of the pipeline
  // Each filter is equally weighted across all iterations of the stack
  itk::ProgressAccumulator::Pointer progress = itk::ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);
  auto frac = (1.0f / 4) / itk::Math::ceil(double(nProj) / m_ProjectionSubsetSize);
  progress->RegisterInternalFilter(m_WeightFilter1, frac);
  progress->RegisterInternalFilter(m_WeightFilter2, frac);
  progress->RegisterInternalFilter(m_VarianceRampFilter, frac);
  progress->RegisterInternalFilter(m_BackProjectionFilter, frac);

  for (unsigned int i = 0; i < nProj; i += m_ProjectionSubsetSize)
  {
    // After the first bp update, we need to use its output as input.
    if (i)
    {
      typename TInputImage::Pointer pimg = m_BackProjectionFilter->GetOutput();
      pimg->DisconnectPipeline();
      m_BackProjectionFilter->SetInput(pimg);

      // Change projection subset
      subsetRegion.SetIndex(Dimension - 1, i);
      subsetRegion.SetSize(Dimension - 1, std::min(m_ProjectionSubsetSize, nProj - i));
      m_ExtractFilter->SetExtractionRegion(subsetRegion);

      // This is required to reset the full pipeline
      m_BackProjectionFilter->GetOutput()->UpdateOutputInformation();
      m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();
    }
    m_BackProjectionFilter->Update();
  }

  this->GraftOutput(m_BackProjectionFilter->GetOutput());
  this->GenerateOutputInformation();
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FDKVarianceReconstructionFilter<TInputImage, TOutputImage, TFFTPrecision>::SetBackProjectionFilter(
  const BackProjectionFilterPointer _arg)
{
  itkDebugMacro("setting BackProjectionFilter to " << _arg);
  if (this->m_BackProjectionFilter != _arg)
  {
    this->m_BackProjectionFilter = _arg;
    m_BackProjectionFilter->SetInput(1, m_VarianceRampFilter->GetOutput());
    this->Modified();
  }
}

} // end namespace rtk

#endif // rtkFDKVarianceReconstructionFilter_hxx
