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

#ifndef rtkSubRegionViewImageFilter_hxx
#define rtkSubRegionViewImageFilter_hxx

namespace rtk
{

template <typename TImage>
void
SubRegionViewImageFilter<TImage>::GenerateInputRequestedRegion()
{
  // The output can span the full extraction region, so ask for it all.
  auto * input = const_cast<TImage *>(this->GetInput());
  if (input)
  {
    input->SetRequestedRegion(m_ExtractionRegion);
  }
}

template <typename TImage>
void
SubRegionViewImageFilter<TImage>::GenerateOutputInformation()
{
  const auto * input = this->GetInput();
  auto *       output = this->GetOutput();

  output->SetRegions(m_ExtractionRegion);
  output->SetSpacing(input->GetSpacing());
  output->SetOrigin(input->GetOrigin());
  output->SetDirection(input->GetDirection());
  output->SetNumberOfComponentsPerPixel(input->GetNumberOfComponentsPerPixel());
}

template <typename TImage>
void
SubRegionViewImageFilter<TImage>::GenerateData()
{
  using PixelType = typename TImage::PixelType;

  const auto * input = this->GetInput();
  auto *       output = this->GetOutput();

  m_IsContiguous = IsContiguousSubRegion(input, m_ExtractionRegion);

  const PixelType * inputPtr = input->GetBufferPointer();
  // Input buffer not available yet: return a metadata-only output.
  if (!inputPtr)
    return;

  if (m_IsContiguous)
  {
    // Zero-copy: share the input pixel buffer, starting at the offset of the
    // region's first pixel (computed over all dimensions so size-1 dimensions
    // are handled: they still contribute their stride to later dimensions).
    const typename TImage::SizeType::SizeValueType offset = ComputePixelOffset(input, m_ExtractionRegion.GetIndex());
    const typename TImage::SizeType::SizeValueType numPixels = m_ExtractionRegion.GetNumberOfPixels();
    output->GetPixelContainer()->SetImportPointer(const_cast<PixelType *>(inputPtr + offset), numPixels, false);

    // Re-assign pixel container to sync subclass containers (e.g.
    // CudaDataManager reads the CPU pointer and marks GPU dirty).
    output->SetPixelContainer(output->GetPixelContainer());
  }
  else
  {
    // Non-contiguous: fall back to itk::ExtractImageFilter (a real copy).
    using ExtractFilterType = itk::ExtractImageFilter<TImage, TImage>;
    typename ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
    extractFilter->SetInput(input);
    extractFilter->SetExtractionRegion(m_ExtractionRegion);
    extractFilter->SetDirectionCollapseToSubmatrix();
    extractFilter->Update();
    output->Graft(extractFilter->GetOutput());
  }
}

} // namespace rtk

#endif // rtkSubRegionViewImageFilter_hxx
