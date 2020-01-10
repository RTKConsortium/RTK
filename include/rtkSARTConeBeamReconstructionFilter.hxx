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

#ifndef rtkSARTConeBeamReconstructionFilter_hxx
#define rtkSARTConeBeamReconstructionFilter_hxx

#include "rtkSARTConeBeamReconstructionFilter.h"

#include <algorithm>
#include <itkIterationReporter.h>

namespace rtk
{
template <class TVolumeImage, class TProjectionImage>
SARTConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::SARTConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set default parameters
  m_EnforcePositivity = false;
  m_IsGated = false;
  m_NumberOfIterations = 3;
  m_Lambda = 0.3;

  // Create each filter of the composite filter
  m_ExtractFilter = ExtractFilterType::New();
  m_ZeroMultiplyFilter = MultiplyFilterType::New();
  m_SubtractFilter = SubtractFilterType::New();
  m_AddFilter = AddFilterType::New();
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_GatingWeightsFilter = GatingWeightsFilterType::New();
  m_ConstantVolumeSource = ConstantVolumeSourceType::New();

  // Create the filters required for correct weighting of the difference
  // projection
  m_ExtractFilterRayBox = ExtractFilterType::New();
  m_RayBoxFilter = RayBoxIntersectionFilterType::New();
  m_DivideProjectionFilter = DivideProjectionFilterType::New();
  m_ConstantProjectionStackSource = ConstantProjectionSourceType::New();

  // Create the filter that enforces positivity
  m_ThresholdFilter = ThresholdFilterType::New();

  // Create the filters required for the normalization of the
  // backprojection
  m_OneConstantProjectionStackSource = ConstantProjectionSourceType::New();
  m_DivideVolumeFilter = DivideVolumeFilterType::New();

  // Permanent internal connections
  m_ZeroMultiplyFilter->SetInput1(itk::NumericTraits<typename ProjectionType::PixelType>::ZeroValue());
  m_ZeroMultiplyFilter->SetInput2(m_ExtractFilter->GetOutput());

  m_SubtractFilter->SetInput(0, m_ExtractFilter->GetOutput());

  m_MultiplyFilter->SetInput1(itk::NumericTraits<typename ProjectionType::PixelType>::ZeroValue());
  m_MultiplyFilter->SetInput2(m_SubtractFilter->GetOutput());

  m_AddFilter->SetInput1(m_DivideVolumeFilter->GetOutput());

  m_ExtractFilterRayBox->SetInput(m_ConstantProjectionStackSource->GetOutput());
  m_RayBoxFilter->SetInput(m_ExtractFilterRayBox->GetOutput());
  m_DivideProjectionFilter->SetInput1(m_MultiplyFilter->GetOutput());
  m_DivideProjectionFilter->SetInput2(m_RayBoxFilter->GetOutput());
  m_DisplacedDetectorFilter->SetInput(m_DivideProjectionFilter->GetOutput());

  // Default parameters
  m_ExtractFilter->SetDirectionCollapseToSubmatrix();
  m_ExtractFilterRayBox->SetDirectionCollapseToSubmatrix();
  m_IsGated = false;
  m_NumberOfProjectionsPerSubset = 1; // Default is the SART behavior
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
  m_DisableDisplacedDetectorFilter = false;
}

template <class TVolumeImage, class TProjectionImage>
void
SARTConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::SetForwardProjectionFilter(ForwardProjectionType _arg)
{
  if (_arg != this->GetForwardProjectionFilter())
  {
    Superclass::SetForwardProjectionFilter(_arg);
    m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter(_arg);
  }
}

template <class TVolumeImage, class TProjectionImage>
void
SARTConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::SetBackProjectionFilter(BackProjectionType _arg)
{
  if (_arg != this->GetBackProjectionFilter())
  {
    Superclass::SetBackProjectionFilter(_arg);
    m_BackProjectionFilter = this->InstantiateBackProjectionFilter(_arg);
    m_BackProjectionNormalizationFilter = this->InstantiateBackProjectionFilter(_arg);
  }
}

template <class TVolumeImage, class TProjectionImage>
void
SARTConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::SetGatingWeights(std::vector<float> weights)
{
  m_GatingWeights = weights;
  m_IsGated = true;
}

template <class TVolumeImage, class TProjectionImage>
void
SARTConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer inputPtr = const_cast<TVolumeImage *>(this->GetInput());

  if (!inputPtr)
    return;

  if (m_EnforcePositivity)
  {
    m_ThresholdFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
    m_ThresholdFilter->GetOutput()->PropagateRequestedRegion();
  }
  else
  {
    m_BackProjectionFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
    m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();
  }
}

template <class TVolumeImage, class TProjectionImage>
void
SARTConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::GenerateOutputInformation()
{
  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);

  // We only set the first sub-stack at that point, the rest will be
  // requested in the GenerateData function
  typename ExtractFilterType::InputImageRegionType projRegion;

  projRegion = this->GetInput(1)->GetLargestPossibleRegion();
  m_ExtractFilter->SetExtractionRegion(projRegion);
  m_ExtractFilterRayBox->SetExtractionRegion(projRegion);

  // Links with the forward and back projection filters should be set here
  // and not in the constructor, as these filters are set at runtime
  m_ConstantVolumeSource->SetInformationFromImage(const_cast<TVolumeImage *>(this->GetInput(0)));
  m_ConstantVolumeSource->SetConstant(0);
  m_ConstantVolumeSource->UpdateOutputInformation();

  m_OneConstantProjectionStackSource->SetInformationFromImage(
    const_cast<TProjectionImage *>(m_ExtractFilter->GetOutput()));
  m_OneConstantProjectionStackSource->SetConstant(1);

  m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_DisplacedDetectorFilter->GetOutput());
  m_BackProjectionFilter->SetTranspose(false);

  m_BackProjectionNormalizationFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionNormalizationFilter->SetInput(1, m_OneConstantProjectionStackSource->GetOutput());
  m_BackProjectionNormalizationFilter->SetTranspose(false);

  m_DivideVolumeFilter->SetInput1(m_BackProjectionFilter->GetOutput());
  m_DivideVolumeFilter->SetInput2(m_BackProjectionNormalizationFilter->GetOutput());
  m_DivideVolumeFilter->SetConstant(0);

  m_AddFilter->SetInput2(this->GetInput(0));

  m_ForwardProjectionFilter->SetInput(0, m_ZeroMultiplyFilter->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, this->GetInput(0));
  m_ExtractFilter->SetInput(this->GetInput(1));
  m_SubtractFilter->SetInput(1, m_ForwardProjectionFilter->GetOutput());

  // For the same reason, set geometry now
  // Check and set geometry
  if (this->GetGeometry() == nullptr)
  {
    itkGenericExceptionMacro(<< "The geometry of the reconstruction has not been set");
  }
  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry);
  m_BackProjectionNormalizationFilter->SetGeometry(this->m_Geometry);
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);

  if (m_IsGated) // For gated SART, insert a gating filter into the pipeline
  {
    m_GatingWeightsFilter->SetInput1(m_DivideProjectionFilter->GetOutput());
    m_GatingWeightsFilter->SetConstant2(1);
    m_DisplacedDetectorFilter->SetInput(m_GatingWeightsFilter->GetOutput());
  }

  m_ConstantProjectionStackSource->SetInformationFromImage(const_cast<TProjectionImage *>(this->GetInput(1)));
  m_ConstantProjectionStackSource->SetConstant(0);
  m_ConstantProjectionStackSource->UpdateOutputInformation();


  // Create the m_RayBoxFiltersectionImageFilter
  m_RayBoxFilter->SetGeometry(this->GetGeometry());

  m_RayBoxFilter->SetBoxFromImage(this->GetInput(0), false);

  if (m_EnforcePositivity)
  {
    m_ThresholdFilter->SetOutsideValue(0);
    m_ThresholdFilter->ThresholdBelow(0);
    m_ThresholdFilter->SetInput(m_AddFilter->GetOutput());

    // Update output information
    m_ThresholdFilter->UpdateOutputInformation();
    this->GetOutput()->SetOrigin(m_ThresholdFilter->GetOutput()->GetOrigin());
    this->GetOutput()->SetSpacing(m_ThresholdFilter->GetOutput()->GetSpacing());
    this->GetOutput()->SetDirection(m_ThresholdFilter->GetOutput()->GetDirection());
    this->GetOutput()->SetLargestPossibleRegion(m_ThresholdFilter->GetOutput()->GetLargestPossibleRegion());
  }
  else
  {
    // Update output information
    m_AddFilter->UpdateOutputInformation();
    this->GetOutput()->SetOrigin(m_AddFilter->GetOutput()->GetOrigin());
    this->GetOutput()->SetSpacing(m_AddFilter->GetOutput()->GetSpacing());
    this->GetOutput()->SetDirection(m_AddFilter->GetOutput()->GetDirection());
    this->GetOutput()->SetLargestPossibleRegion(m_AddFilter->GetOutput()->GetLargestPossibleRegion());
  }

  // Set memory management flags
  m_ZeroMultiplyFilter->ReleaseDataFlagOn();
  m_ForwardProjectionFilter->ReleaseDataFlagOn();
  m_SubtractFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_RayBoxFilter->ReleaseDataFlagOn();
  m_DivideProjectionFilter->ReleaseDataFlagOn();
  m_DisplacedDetectorFilter->ReleaseDataFlagOn();
  m_DivideVolumeFilter->ReleaseDataFlagOn();

  if (m_EnforcePositivity)
    m_AddFilter->ReleaseDataFlagOn();
}

template <class TVolumeImage, class TProjectionImage>
void
SARTConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::GenerateData()
{
  const unsigned int Dimension = this->InputImageDimension;

  // The backprojection works on one projection at a time
  typename ExtractFilterType::InputImageRegionType subsetRegion;
  subsetRegion = this->GetInput(1)->GetLargestPossibleRegion();
  unsigned int nProj = subsetRegion.GetSize(Dimension - 1);
  subsetRegion.SetSize(Dimension - 1, 1);

  // Fill and shuffle randomly the projection order.
  // Should be tunable with other solutions.
  std::vector<unsigned int> projOrder(nProj);

  for (unsigned int i = 0; i < nProj; i++)
    projOrder[i] = i;
  std::shuffle(projOrder.begin(), projOrder.end(), Superclass::m_DefaultRandomEngine);

  m_MultiplyFilter->SetInput1((const float)m_Lambda);

  // Create the zero projection stack used as input by RayBoxIntersectionFilter
  m_ConstantProjectionStackSource->Update();

  // Declare the image used in the main loop
  typename TVolumeImage::Pointer pimg;
  typename TVolumeImage::Pointer norm;

  itk::IterationReporter iterationReporter(this, 0, 1); // report every iteration

  // For each iteration, go over each projection
  for (unsigned int iter = 0; iter < m_NumberOfIterations; iter++)
  {
    unsigned int projectionsProcessedInSubset = 0;
    for (unsigned int i = 0; i < nProj; i++)
    {
      // Change projection subset
      subsetRegion.SetIndex(Dimension - 1, projOrder[i]);
      m_ExtractFilter->SetExtractionRegion(subsetRegion);
      m_ExtractFilterRayBox->SetExtractionRegion(subsetRegion);
      m_ExtractFilter->UpdateOutputInformation();

      m_OneConstantProjectionStackSource->SetInformationFromImage(
        const_cast<TProjectionImage *>(m_ExtractFilter->GetOutput()));

      // Set gating weight for the current projection
      if (m_IsGated)
      {
        m_GatingWeightsFilter->SetConstant2(m_GatingWeights[i]);
      }

      // This is required to reset the full pipeline
      m_BackProjectionFilter->GetOutput()->UpdateOutputInformation();
      m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();
      m_BackProjectionNormalizationFilter->GetOutput()->UpdateOutputInformation();
      m_BackProjectionNormalizationFilter->GetOutput()->PropagateRequestedRegion();

      projectionsProcessedInSubset++;
      if ((projectionsProcessedInSubset == m_NumberOfProjectionsPerSubset) || (i == nProj - 1))
      {
        m_DivideVolumeFilter->SetInput2(m_BackProjectionNormalizationFilter->GetOutput());
        m_DivideVolumeFilter->SetInput1(m_BackProjectionFilter->GetOutput());
        m_AddFilter->SetInput1(m_DivideVolumeFilter->GetOutput());
        m_DivideVolumeFilter->Update();

        // To start a new subset:
        // - plug the output of the pipeline back into the Forward projection filter
        // - set the input of the Back projection filter to zero
        if (m_EnforcePositivity)
          pimg = m_ThresholdFilter->GetOutput();
        else
          pimg = m_AddFilter->GetOutput();
        pimg->Update();
        pimg->DisconnectPipeline();

        m_ForwardProjectionFilter->SetInput(1, pimg);
        m_AddFilter->SetInput2(pimg);
        m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
        m_BackProjectionNormalizationFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());

        projectionsProcessedInSubset = 0;
      }
      // Backproject in the same image otherwise.
      else
      {
        m_BackProjectionFilter->Update();
        m_BackProjectionNormalizationFilter->Update();
        pimg = m_BackProjectionFilter->GetOutput();
        pimg->DisconnectPipeline();
        m_BackProjectionFilter->SetInput(0, pimg);
        norm = m_BackProjectionNormalizationFilter->GetOutput();
        norm->DisconnectPipeline();
        m_BackProjectionNormalizationFilter->SetInput(0, norm);
      }
    }
    this->GraftOutput(pimg);
    iterationReporter.CompletedStep();
  }
}

} // end namespace rtk

#endif // rtkSARTConeBeamReconstructionFilter_hxx
