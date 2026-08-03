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

#ifndef rtkExtractImageSubRegion_h
#define rtkExtractImageSubRegion_h

#include <itkImage.h>
#include <itkImageRegion.h>
#include <itkExtractImageFilter.h>

namespace rtk
{

/** \class ExtractImageSubRegion
 * \brief Create an image that is a view of a sub-region of another image,
 *        without copying pixel data when the region is contiguous.
 *
 * This is a lightweight alternative to itk::ExtractImageFilter for the case
 * where input and output image types are the same dimension (no dimension
 * collapse). It avoids the overhead of the filter pipeline machinery by
 * directly creating an image that shares the same pixel buffer as the input,
 * with adjusted metadata (origin, region).
 *
 * The zero-copy optimization only applies when the extraction region is
 * contiguous in memory, i.e. all dimensions except the last span the full
 * input extent. When the region is not contiguous, falls back to
 * itk::ExtractImageFilter.
 *
 * When the input buffer is not yet allocated (e.g., during
 * GenerateOutputInformation), the output image is created with the correct
 * metadata only. When the buffer is available (e.g., during GenerateData),
 * the pixel buffer is shared via SetImportPointer for zero-copy access.
 *
 * Warning: since the output shares the input's pixel buffer, downstream
 * filters that operate in-place (InPlaceOn) will corrupt the source data.
 * Callers must ensure that no in-place filter modifies this image's buffer.
 *
 * This is useful in mini-pipelines where a sub-stack of projections is
 * repeatedly extracted from a projection stack (e.g., FDK, SART, OSEM).
 *
 * \author Axel Garcia
 *
 * \ingroup RTK
 */
/** Check if a sub-region is contiguous in the input buffer (zero-copy possible).
 *  True when all non-last dimensions match the input exactly. */
template <typename TImage>
bool
IsContiguousSubRegion(const TImage * input, const itk::ImageRegion<TImage::ImageDimension> & region)
{
  constexpr unsigned int Dimension = TImage::ImageDimension;
  const auto &           inputRegion = input->GetLargestPossibleRegion();
  for (unsigned int d = 0; d < Dimension - 1; ++d)
  {
    if (region.GetIndex()[d] != inputRegion.GetIndex()[d] || region.GetSize()[d] != inputRegion.GetSize()[d])
      return false;
  }
  return true;
}

template <typename TImage>
typename TImage::Pointer
ExtractImageSubRegion(const TImage * input, const itk::ImageRegion<TImage::ImageDimension> & extractionRegion)
{
  constexpr unsigned int Dimension = TImage::ImageDimension;
  using PixelType = typename TImage::PixelType;
  using RegionType = itk::ImageRegion<Dimension>;
  using SizeType = itk::Size<Dimension>;
  using IndexType = itk::Index<Dimension>;
  using SpacingType = typename TImage::SpacingType;
  using PointType = typename TImage::PointType;
  using DirectionType = typename TImage::DirectionType;

  const RegionType & inputRegion = input->GetLargestPossibleRegion();
  const IndexType &  inputIndex = inputRegion.GetIndex();

  if (!IsContiguousSubRegion(input, extractionRegion))
  {
    using ExtractFilterType = itk::ExtractImageFilter<TImage, TImage>;
    typename ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
    extractFilter->SetInput(input);
    extractFilter->SetExtractionRegion(extractionRegion);
    extractFilter->SetDirectionCollapseToSubmatrix();
    extractFilter->Update();
    return extractFilter->GetOutput();
  }

  const SpacingType &   spacing = input->GetSpacing();
  const PointType &     inputOrigin = input->GetOrigin();
  const DirectionType & direction = input->GetDirection();

  // Create output with correct metadata (skips Allocate for CudaImage).
  typename TImage::Pointer output = TImage::New();
  output->SetRegions(extractionRegion);
  output->SetSpacing(spacing);
  output->SetOrigin(inputOrigin);
  output->SetDirection(direction);

  // If the input buffer is available, share it (zero-copy).
  // Otherwise, return a metadata-only image.
  if (input->GetBufferPointer())
  {
    const IndexType & extractIndex = extractionRegion.GetIndex();

    // Pixels per slice: product of all input sizes except the last.
    typename SizeType::SizeValueType sliceSize = 1;
    for (unsigned int d = 0; d < Dimension - 1; ++d)
      sliceSize *= inputRegion.GetSize()[d];

    const long        sliceOffset = extractIndex[Dimension - 1] - inputIndex[Dimension - 1];
    const PixelType * bufferPtr = input->GetBufferPointer() + sliceOffset * sliceSize;

    const typename SizeType::SizeValueType numPixels = extractionRegion.GetNumberOfPixels();
    output->GetPixelContainer()->SetImportPointer(const_cast<PixelType *>(bufferPtr), numPixels, false);

    // Re-assign pixel container to sync subclass containers (e.g.
    // CudaDataManager reads the CPU pointer and marks GPU dirty).
    output->SetPixelContainer(output->GetPixelContainer());
  }

  return output;
}

} // namespace rtk

#endif // rtkExtractImageSubRegion_h
