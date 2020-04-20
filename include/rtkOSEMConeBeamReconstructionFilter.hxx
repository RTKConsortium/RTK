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

#ifndef rtkOSEMConeBeamReconstructionFilter_hxx
#define rtkOSEMConeBeamReconstructionFilter_hxx

#include "rtkOSEMConeBeamReconstructionFilter.h"
#include "rtkGeneralPurposeFunctions.h"

#include <algorithm>
#include <itkTimeProbe.h>

#include <itkImageFileWriter.h>

#include <itkIterationReporter.h>

namespace rtk
{
template <class TVolumeImage, class TProjectionImage>
OSEMConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::OSEMConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Create each filter of the composite filter
  m_ExtractFilter = ExtractFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_ConstantVolumeSource = ConstantVolumeSourceType::New();
  m_ZeroConstantProjectionStackSource = ConstantProjectionSourceType::New();
  m_DivideProjectionFilter = DivideProjectionFilterType::New();
  m_DePierroRegularizationFilter = DePierroRegularizationFilterType::New();

  // Create the filters required for the normalization of the
  // backprojection
  m_OneConstantProjectionStackSource = ConstantProjectionSourceType::New();
  m_DivideVolumeFilter = DivideVolumeFilterType::New();

  // Permanent internal connections
  m_DivideProjectionFilter->SetInput1(m_ExtractFilter->GetOutput());
  m_DivideVolumeFilter->SetInput1(m_MultiplyFilter->GetOutput());

  // Default parameters
  m_ExtractFilter->SetDirectionCollapseToSubmatrix();
}

template <class TVolumeImage, class TProjectionImage>
void
OSEMConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::VerifyPreconditions() ITKv5_CONST
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TVolumeImage, class TProjectionImage>
void
OSEMConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer inputPtr = const_cast<TVolumeImage *>(this->GetInput());

  if (!inputPtr)
    return;

  m_BackProjectionFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
  m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();
}

template <class TVolumeImage, class TProjectionImage>
void
OSEMConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::GenerateOutputInformation()
{

  // We only set the first sub-stack at that point, the rest will be
  // requested in the GenerateData function
  typename ExtractFilterType::InputImageRegionType projRegion;

  // Set forward projection filter
  m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter(this->m_CurrentForwardProjectionConfiguration);

  // Set back projection filter
  m_BackProjectionFilter = this->InstantiateBackProjectionFilter(this->m_CurrentBackProjectionConfiguration);
  m_BackProjectionNormalizationFilter =
    this->InstantiateBackProjectionFilter(this->m_CurrentBackProjectionConfiguration);

  projRegion = this->GetInput(1)->GetLargestPossibleRegion();
  m_ExtractFilter->SetExtractionRegion(projRegion);

  m_ExtractFilter->SetInput(this->GetInput(1));
  m_ExtractFilter->UpdateOutputInformation();

  // Links with the forward and back projection filters should be set here
  // and not in the constructor, as these filters are set at runtime
  m_ConstantVolumeSource->SetInformationFromImage(const_cast<TVolumeImage *>(this->GetInput(0)));
  m_ConstantVolumeSource->SetConstant(0);

  m_OneConstantProjectionStackSource->SetInformationFromImage(
    const_cast<TProjectionImage *>(m_ExtractFilter->GetOutput()));
  m_OneConstantProjectionStackSource->SetConstant(1);

  m_ZeroConstantProjectionStackSource->SetInformationFromImage(
    const_cast<TProjectionImage *>(m_ExtractFilter->GetOutput()));
  m_ZeroConstantProjectionStackSource->SetConstant(0);

  m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_DivideProjectionFilter->GetOutput());
  if (this->GetBackProjectionFilter() == this->BP_JOSEPHATTENUATED || this->GetBackProjectionFilter() == this->BP_ZENG)
  {
    if (!(this->GetInput(2)) && this->GetBackProjectionFilter() == this->BP_JOSEPHATTENUATED)
    {
      itkExceptionMacro(<< "Set Joseph attenuated backprojection filter but no attenuation map is given");
    }
    else
    {
      m_BackProjectionFilter->SetInput(2, this->GetInput(2));
    }
  }
  if (m_SigmaZero != -1 || m_Alpha != -1)
  {
    if (this->GetBackProjectionFilter() == this->BP_ZENG)
    {
      if (m_SigmaZero != -1)
        dynamic_cast<ZengBackProjectionImageFilter<TVolumeImage, TProjectionImage> *>(
          m_BackProjectionFilter.GetPointer())
          ->SetSigmaZero(m_SigmaZero);
      if (m_Alpha != -1)
        dynamic_cast<ZengBackProjectionImageFilter<TVolumeImage, TProjectionImage> *>(
          m_BackProjectionFilter.GetPointer())
          ->SetAlpha(m_Alpha);
    }
    else
      itkExceptionMacro(<< "PSF correction only available with Zeng projector type");
  }

  m_BackProjectionFilter->SetTranspose(false);

  m_BackProjectionNormalizationFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionNormalizationFilter->SetInput(1, m_OneConstantProjectionStackSource->GetOutput());
  if (this->GetBackProjectionFilter() == this->BP_JOSEPHATTENUATED || this->GetBackProjectionFilter() == this->BP_ZENG)
  {
    if (!(this->GetInput(2)) && this->GetBackProjectionFilter() == this->BP_JOSEPHATTENUATED)
    {
      itkExceptionMacro(<< "Set Joseph attenuated backprojection filter but no attenuation map is given");
    }
    else
    {
      m_BackProjectionNormalizationFilter->SetInput(2, this->GetInput(2));
    }
  }
  if (m_SigmaZero != -1 || m_Alpha != -1)
  {
    if (this->GetBackProjectionFilter() == this->BP_ZENG)
    {
      if (m_SigmaZero != -1)
        dynamic_cast<ZengBackProjectionImageFilter<TVolumeImage, TProjectionImage> *>(
          m_BackProjectionNormalizationFilter.GetPointer())
          ->SetSigmaZero(m_SigmaZero);
      if (m_Alpha != -1)
        dynamic_cast<ZengBackProjectionImageFilter<TVolumeImage, TProjectionImage> *>(
          m_BackProjectionNormalizationFilter.GetPointer())
          ->SetAlpha(m_Alpha);
    }
    else
      itkExceptionMacro(<< "PSF correction only available with Zeng projector type");
  }

  m_BackProjectionNormalizationFilter->SetTranspose(false);

  m_MultiplyFilter->SetInput1(m_BackProjectionFilter->GetOutput());
  m_MultiplyFilter->SetInput2(this->GetInput(0));

  m_DePierroRegularizationFilter->SetInput(0, this->GetInput(0));
  m_DePierroRegularizationFilter->SetInput(1, m_MultiplyFilter->GetOutput());
  m_DePierroRegularizationFilter->SetInput(2, m_BackProjectionNormalizationFilter->GetOutput());
  m_DePierroRegularizationFilter->SetBeta(m_BetaRegularization);
  m_DivideVolumeFilter->SetInput2(m_DePierroRegularizationFilter->GetOutput());
  m_DivideVolumeFilter->SetConstant(0);

  m_ForwardProjectionFilter->SetInput(0, m_ZeroConstantProjectionStackSource->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, this->GetInput(0));
  if (this->GetForwardProjectionFilter() == this->FP_JOSEPHATTENUATED ||
      this->GetForwardProjectionFilter() == this->FP_ZENG)
  {
    if (!(this->GetInput(2)) && this->GetForwardProjectionFilter() == this->FP_JOSEPHATTENUATED)
    {
      itkExceptionMacro(<< "Set Joseph attenuated forward projection filter but no attenuation map is given");
    }
    else
    {
      m_ForwardProjectionFilter->SetInput(2, this->GetInput(2));
    }
  }
  if (m_SigmaZero != -1 || m_Alpha != -1)
  {
    if (this->GetForwardProjectionFilter() == this->FP_ZENG)
    {
      if (m_SigmaZero != -1)
        dynamic_cast<ZengForwardProjectionImageFilter<TProjectionImage, TVolumeImage> *>(
          m_ForwardProjectionFilter.GetPointer())
          ->SetSigmaZero(m_SigmaZero);
      if (m_Alpha != -1)
        dynamic_cast<ZengForwardProjectionImageFilter<TProjectionImage, TVolumeImage> *>(
          m_ForwardProjectionFilter.GetPointer())
          ->SetAlpha(m_Alpha);
    }
    else
      itkExceptionMacro(<< "PSF correction only available with Zeng projector type");
  }

  m_DivideProjectionFilter->SetInput2(m_ForwardProjectionFilter->GetOutput());
  m_DivideProjectionFilter->SetConstant(1);

  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry);
  m_BackProjectionNormalizationFilter->SetGeometry(this->m_Geometry);

  // Update output information
  m_DivideVolumeFilter->UpdateOutputInformation();
  this->GetOutput()->SetOrigin(m_DivideVolumeFilter->GetOutput()->GetOrigin());
  this->GetOutput()->SetSpacing(m_DivideVolumeFilter->GetOutput()->GetSpacing());
  this->GetOutput()->SetDirection(m_DivideVolumeFilter->GetOutput()->GetDirection());
  this->GetOutput()->SetLargestPossibleRegion(m_DivideVolumeFilter->GetOutput()->GetLargestPossibleRegion());

  // Set memory management flags
  m_ForwardProjectionFilter->ReleaseDataFlagOn();
  m_DivideProjectionFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
}

template <class TVolumeImage, class TProjectionImage>
void
OSEMConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>::GenerateData()
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

  // If m_StoreNormalizationImages is true, the backprojection of
  // ones will be only computed once during the first iteration.
  // The result will be stored in an vector and reused for the next iterations.
  std::vector<typename TVolumeImage::Pointer> vectorNorm;

  for (unsigned int i = 0; i < nProj; i++)
    projOrder[i] = i;
  std::shuffle(projOrder.begin(), projOrder.end(), Superclass::m_DefaultRandomEngine);

  // Declare the image used in the main loop
  typename TVolumeImage::Pointer pimg;
  typename TVolumeImage::Pointer norm;

  itk::IterationReporter iterationReporter(this, 0, 1);

  // For each iteration, go over each projection
  for (unsigned int iter = 0; iter < m_NumberOfIterations; iter++)
  {
    unsigned int projectionsProcessedInSubset = 0;
    unsigned int currentSubset = 0;
    for (unsigned int i = 0; i < nProj; i++)
    {
      // Change projection subset
      subsetRegion.SetIndex(Dimension - 1, projOrder[i]);
      m_ExtractFilter->SetExtractionRegion(subsetRegion);
      m_ExtractFilter->UpdateOutputInformation();

      m_ZeroConstantProjectionStackSource->SetInformationFromImage(
        const_cast<TProjectionImage *>(m_ExtractFilter->GetOutput()));

      // This is required to reset the full pipeline
      m_BackProjectionFilter->GetOutput()->UpdateOutputInformation();
      m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();

      m_BackProjectionFilter->Update();
      if (iter == 0 || !m_StoreNormalizationImages)
      {
        m_OneConstantProjectionStackSource->SetInformationFromImage(
          const_cast<TProjectionImage *>(m_ExtractFilter->GetOutput()));
        m_BackProjectionNormalizationFilter->GetOutput()->UpdateOutputInformation();
        m_BackProjectionNormalizationFilter->GetOutput()->PropagateRequestedRegion();
        m_BackProjectionNormalizationFilter->Update();
      }

      projectionsProcessedInSubset++;
      if ((projectionsProcessedInSubset == m_NumberOfProjectionsPerSubset) || (i == nProj - 1))
      {
        if (iter == 0 && m_StoreNormalizationImages)
        {
          vectorNorm.push_back(m_BackProjectionNormalizationFilter->GetOutput());
          vectorNorm.back()->DisconnectPipeline();
        }
        m_MultiplyFilter->SetInput1(m_BackProjectionFilter->GetOutput());

        m_DivideVolumeFilter->SetInput1(m_MultiplyFilter->GetOutput());
        m_DePierroRegularizationFilter->SetInput(1, m_MultiplyFilter->GetOutput());
        if (m_StoreNormalizationImages)
          m_DePierroRegularizationFilter->SetInput(2, vectorNorm[currentSubset]);
        else
          m_DePierroRegularizationFilter->SetInput(2, m_BackProjectionNormalizationFilter->GetOutput());
        m_DePierroRegularizationFilter->Update();

        m_DivideVolumeFilter->SetInput2(m_DePierroRegularizationFilter->GetOutput());
        m_DivideVolumeFilter->Update();

        // To start a new subset:
        // - plug the output of the pipeline back into the Forward projection filter
        // - set the input of the Back projection filter to zero
        pimg = m_DivideVolumeFilter->GetOutput();
        pimg->DisconnectPipeline();

        m_ForwardProjectionFilter->SetInput(1, pimg);
        m_DePierroRegularizationFilter->SetInput(0, pimg);
        m_MultiplyFilter->SetInput2(pimg);
        m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
        m_BackProjectionNormalizationFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());

        currentSubset++;
        projectionsProcessedInSubset = 0;
      }
      // Backproject in the same image otherwise.
      else
      {
        pimg = m_BackProjectionFilter->GetOutput();
        pimg->DisconnectPipeline();
        m_BackProjectionFilter->SetInput(0, pimg);
        if (iter == 0 || !m_StoreNormalizationImages)
        {
          norm = m_BackProjectionNormalizationFilter->GetOutput();
          norm->DisconnectPipeline();
          m_BackProjectionNormalizationFilter->SetInput(0, norm);
        }
      }
    }
    this->GraftOutput(pimg);
    iterationReporter.CompletedStep();
  }
  vectorNorm.clear();
}

} // end namespace rtk

#endif // rtkOSEMConeBeamReconstructionFilter_hxx
