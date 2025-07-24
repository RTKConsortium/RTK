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
#ifndef rtkWarpProjectionStackToFourDImageFilter_hxx
#define rtkWarpProjectionStackToFourDImageFilter_hxx


#include <itkObjectFactory.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

namespace rtk
{

template <typename VolumeSeriesType, typename ProjectionStackType>
WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>::WarpProjectionStackToFourDImageFilter()
{
  this->SetNumberOfRequiredInputs(3);

  this->m_UseCudaSplat = !std::is_same_v<VolumeSeriesType, CPUVolumeSeriesType>;
  this->m_UseCudaSources = !std::is_same_v<VolumeSeriesType, CPUVolumeSeriesType>;
  m_UseCudaCyclicDeformation = false;

  this->m_BackProjectionFilter = WarpBackProjectionImageFilter::New();
  if (std::is_same_v<VolumeSeriesType, CPUVolumeSeriesType>)
    itkWarningMacro("The warp back project image filter exists only in CUDA. Ignoring the displacement vector field "
                    "and using CPU voxel-based back projection");
}

template <typename VolumeSeriesType, typename ProjectionStackType>
void
WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>::SetDisplacementField(
  const DVFSequenceImageType * DisplacementField)
{
  this->SetNthInput(2, const_cast<DVFSequenceImageType *>(DisplacementField));
}

template <typename VolumeSeriesType, typename ProjectionStackType>
typename WarpProjectionStackToFourDImageFilter<VolumeSeriesType,
                                               ProjectionStackType>::DVFSequenceImageType::ConstPointer
WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>::GetDisplacementField()
{
  return static_cast<const DVFSequenceImageType *>(this->itk::ProcessObject::GetInput(2));
}

template <typename VolumeSeriesType, typename ProjectionStackType>
void
WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>::SetSignal(
  const std::vector<double> signal)
{
  this->m_Signal = signal;
  this->Modified();
}

template <typename VolumeSeriesType, typename ProjectionStackType>
void
WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>::GenerateOutputInformation()
{
  m_DVFInterpolatorFilter = CPUDVFInterpolatorType::New();
  if (m_UseCudaCyclicDeformation)
  {
    if (std::is_same_v<VolumeSeriesType, CPUVolumeSeriesType>)
      itkGenericExceptionMacro(<< "UseCudaCyclicDeformation option only available with itk::CudaImage.");
    m_DVFInterpolatorFilter = CudaCyclicDeformationImageFilterType::New();
  }
#ifdef RTK_USE_CUDA
  if (!std::is_same_v<VolumeSeriesType, CPUVolumeSeriesType>)
  {
    CudaWarpBackProjectionImageFilter * wbp;
    wbp = dynamic_cast<CudaWarpBackProjectionImageFilter *>(this->m_BackProjectionFilter.GetPointer());
    using CudaDVFImageType = itk::CudaImage<VectorForDVF, VolumeSeriesType::ImageDimension - 1>;
    CudaDVFImageType * cudvf;
    cudvf = dynamic_cast<CudaDVFImageType *>(m_DVFInterpolatorFilter->GetOutput());
    wbp->SetDisplacementField(cudvf);
  }
#endif
  m_DVFInterpolatorFilter->SetSignalVector(m_Signal);
  m_DVFInterpolatorFilter->SetInput(this->GetDisplacementField());
  m_DVFInterpolatorFilter->SetFrame(0);

  Superclass::GenerateOutputInformation();
}

template <typename VolumeSeriesType, typename ProjectionStackType>
void
WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>::GenerateData()
{
  int Dimension = ProjectionStackType::ImageDimension;

  // Set the Extract filter
  typename ProjectionStackType::RegionType extractRegion;
  extractRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  extractRegion.SetSize(Dimension - 1, 1);

  // Declare the pointer to a VolumeSeries that will be used in the pipeline
  typename VolumeSeriesType::Pointer pimg;

  int NumberProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension - 1);
  int FirstProj = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex(Dimension - 1);

  bool firstProjectionProcessed = false;

  // Process the projections in order
  for (int proj = FirstProj; proj < FirstProj + NumberProjs; proj++)
  {
    // After the first update, we need to use the output as input.
    if (firstProjectionProcessed)
    {
      pimg = this->m_SplatFilter->GetOutput();
      pimg->DisconnectPipeline();
      this->m_SplatFilter->SetInputVolumeSeries(pimg);
    }

    // Set the Extract Filter
    extractRegion.SetIndex(Dimension - 1, proj);
    this->m_ExtractFilter->SetExtractionRegion(extractRegion);

    // Set the DVF interpolator
    m_DVFInterpolatorFilter->SetFrame(proj);

    // Set the splat filter
    this->m_SplatFilter->SetProjectionNumber(proj);

    // Update the last filter
    this->m_SplatFilter->Update();

    // Update condition
    firstProjectionProcessed = true;
  }

  // Graft its output
  this->GraftOutput(this->m_SplatFilter->GetOutput());

  // Release the data in internal filters
  if (pimg.IsNotNull())
    pimg->ReleaseData();

  this->m_BackProjectionFilter->GetOutput()->ReleaseData();
  this->m_ExtractFilter->GetOutput()->ReleaseData();
  this->m_ConstantVolumeSource->GetOutput()->ReleaseData();
  this->m_DVFInterpolatorFilter->GetOutput()->ReleaseData();
}

} // namespace rtk


#endif
