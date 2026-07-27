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

#ifndef rtkSubRegionViewImageFilter_h
#define rtkSubRegionViewImageFilter_h

#include <itkIndex.h>
#include <itkImageRegion.h>
#include <itkImageToImageFilter.h>
#include <itkExtractImageFilter.h>

namespace rtk
{

/** Return the row-major element offset of a pixel index within the input's
 *  buffered pixel buffer. */
template <typename TImage>
typename TImage::SizeType::SizeValueType
ComputePixelOffset(const TImage * input, const itk::Index<TImage::ImageDimension> & index)
{
  constexpr unsigned int Dimension = TImage::ImageDimension;
  const auto &           inputRegion = input->GetBufferedRegion();
  const auto &           inputIndex = inputRegion.GetIndex();

  typename TImage::SizeType::SizeValueType stride = 1;
  typename TImage::SizeType::SizeValueType offset = 0;
  for (unsigned int d = 0; d < Dimension; ++d)
  {
    offset += (index[d] - inputIndex[d]) * stride;
    stride *= inputRegion.GetSize()[d];
  }
  return offset;
}

/** Check if a sub-region is a contiguous block of the input pixel buffer
 *  (i.e. its pixels occupy consecutive memory addresses, so it can be viewed
 *  without copying). A dimension of size 1 is handled automatically: it is
 *  contiguous whenever it does not introduce a memory gap. */
template <typename TImage>
bool
IsContiguousSubRegion(const TImage * input, const itk::ImageRegion<TImage::ImageDimension> & region)
{
  // The region must lie inside the buffered region.
  if (!input->GetBufferedRegion().IsInside(region))
    return false;

  // Row-major addresses (relative to the input buffer) of the region's first
  // and last pixels. The region is contiguous iff its pixels span exactly
  // their number of memory addresses between those two addresses.
  itk::Index<TImage::ImageDimension> lastIndex = region.GetIndex();
  for (unsigned int d = 0; d < TImage::ImageDimension; ++d)
    lastIndex[d] += region.GetSize()[d] - 1;

  return ComputePixelOffset(input, lastIndex) - ComputePixelOffset(input, region.GetIndex()) + 1 ==
         region.GetNumberOfPixels();
}

/** \class SubRegionViewImageFilter
 * \brief Extract a sub-region of an image, sharing the buffer when contiguous.
 *
 * The output is a non-owning "view" of a sub-region of the input image. When
 * the extraction region is contiguous (its pixels occupy consecutive memory
 * addresses), the output shares the input pixel buffer without copying pixels
 * (like a numpy view). Otherwise, a copy is made with itk::ExtractImageFilter.
 *
 * This filter derives from itk::ImageToImageFilter, not itk::InPlaceImageFilter,
 * because it is never an in-place filter: it neither modifies nor consumes its
 * input. itk::InPlaceImageFilter would be a poor fit for two reasons: it only
 * shares the buffer when the input's buffered region exactly matches the
 * output's requested region, which never holds for a sub-region (so it would
 * silently fall back to a plain copy), and its semantics are destructive — it
 * grafts the whole input onto the output region and releases the input's data
 * afterwards. Deriving from itk::ImageToImageFilter instead keeps the input
 * intact and usable by other consumers while the output is a smaller view of it.
 *
 * \warning The output shares the input pixel buffer, so downstream filters
 * that operate in place (InPlaceOn) will corrupt the input data. Only use
 * this filter in pipelines without in-place consumers (or call InPlaceOff()
 * on the downstream filters).
 *
 * \author Axel Garcia
 *
 * \ingroup RTK
 */
template <typename TImage>
class ITK_TEMPLATE_EXPORT SubRegionViewImageFilter : public itk::ImageToImageFilter<TImage, TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SubRegionViewImageFilter);

  /** Standard class type alias. */
  using Self = SubRegionViewImageFilter;
  using Superclass = itk::ImageToImageFilter<TImage, TImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using RegionType = itk::ImageRegion<TImage::ImageDimension>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(SubRegionViewImageFilter);

  /** Set the region to extract from the input image. */
  void
  SetExtractionRegion(const RegionType & region)
  {
    if (m_ExtractionRegion != region)
    {
      m_ExtractionRegion = region;
      this->Modified();
    }
  }
  itkGetConstReferenceMacro(ExtractionRegion, RegionType);

  /** After Update(), returns true if the output shares the input buffer. */
  itkGetMacro(IsContiguous, bool);

protected:
  SubRegionViewImageFilter() = default;
  ~SubRegionViewImageFilter() override = default;

  void
  GenerateInputRequestedRegion() override;

  void
  GenerateOutputInformation() override;

  void
  GenerateData() override;

private:
  RegionType m_ExtractionRegion;
  bool       m_IsContiguous{ false };
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSubRegionViewImageFilter.hxx"
#endif

#endif // rtkSubRegionViewImageFilter_h
