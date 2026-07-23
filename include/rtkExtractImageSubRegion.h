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

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif

namespace rtk
{

/** \class ExtractImageSubRegion
 * \brief Create an image that is a view of a sub-region of another image,
 *        without copying pixel data.
 *
 * This is a lightweight alternative to itk::ExtractImageFilter for the case
 * where input and output image types are the same dimension (no dimension
 * collapse). It avoids the overhead of the filter pipeline machinery by
 * directly creating an image that shares the same pixel buffer as the input,
 * with adjusted metadata (origin, region).
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
  const SizeType &   inputSize = inputRegion.GetSize();

  const SpacingType &   spacing = input->GetSpacing();
  const PointType &     inputOrigin = input->GetOrigin();
  const DirectionType & direction = input->GetDirection();

  // Create the output image with correct metadata
  typename TImage::Pointer output = TImage::New();
  output->SetRegions(extractionRegion);
  output->SetSpacing(spacing);
  output->SetOrigin(inputOrigin);
  output->SetDirection(direction);

  // If the input buffer is available, share it (zero-copy).
  // If not (e.g. during GenerateOutputInformation before pipeline execution),
  // just create the metadata-only image. Downstream filters that call
  // UpdateOutputInformation only need the metadata, not the pixels.
  if (input->GetBufferPointer())
  {
    const IndexType & extractIndex = extractionRegion.GetIndex();
    const IndexType & inputIndex = inputRegion.GetIndex();

    // Compute the number of pixels per slice (product of all dimensions except the last)
    typename SizeType::SizeValueType sliceSize = 1;
    for (unsigned int d = 0; d < Dimension - 1; ++d)
    {
      sliceSize *= inputSize[d];
    }

    // Compute the buffer offset to the start of the extraction region
    const long        sliceOffset = extractIndex[Dimension - 1] - inputIndex[Dimension - 1];
    const PixelType * bufferPtr = input->GetBufferPointer() + sliceOffset * sliceSize;

    const typename SizeType::SizeValueType numPixels = extractionRegion.GetNumberOfPixels();
    output->GetPixelContainer()->SetImportPointer(const_cast<PixelType *>(bufferPtr), numPixels, false);

#ifdef RTK_USE_CUDA
    using TCudaImage = itk::CudaImage<PixelType, Dimension>;
    if (TCudaImage * cudaOutput = dynamic_cast<TCudaImage *>(output.GetPointer()))
    {
      cudaOutput->GetModifiableDataManager()->SetBufferSize(numPixels * sizeof(PixelType));
      cudaOutput->GetModifiableDataManager()->SetImagePointer(cudaOutput);
      cudaOutput->GetModifiableDataManager()->SetCPUBufferPointer(const_cast<PixelType *>(bufferPtr));
      cudaOutput->GetModifiableDataManager()->SetGPUDirtyFlag(true);
      cudaOutput->GetModifiableDataManager()->SetCPUDirtyFlag(false);
    }
#endif
  }

  return output;
}

} // namespace rtk

#endif // rtkExtractImageSubRegion_h
