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

#ifndef rtkDownsampleImageFilter_h
#define rtkDownsampleImageFilter_h

#include "itkImageToImageFilter.h"

namespace rtk
{

/** \class DownsampleImageFilter
 * \brief Downsamples an image by a factor in each dimension.
 *
 * This filter is inspired from Dan Mueller's GIFT package
 * http://www.insight-journal.org/browse/publication/103
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT DownsampleImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(DownsampleImageFilter);

  /** Standard class type alias. */
  using Self = DownsampleImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DownsampleImageFilter, ImageToImageFilter);

  /** Typedef to images */
  using OutputImageType = TOutputImage;
  using InputImageType = TInputImage;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using InputImagePointer = typename InputImageType::Pointer;
  using InputImageConstPointer = typename InputImageType::ConstPointer;

  /** Typedef to describe the output image region type. */
  using OutputImageRegionType = typename TOutputImage::RegionType;

  /** ImageDimension enumeration. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Set the downsample factors. Values are clamped to
   * a minimum value of 1.*/
  void
  SetFactors(unsigned int factors[]);

  /** Sets the downsample factor for the given dimension.
   * All other dimensions are set to 1 */
  void
  SetFactor(unsigned int dimension, unsigned int factor);

  /** DownsampleImageFilter produces an image which is a different
   * resolution and with a different pixel spacing than its input
   * image.  As such, DownsampleImageFilter needs to provide an
   * implementation for GenerateOutputInformation() in order to inform
   * the pipeline execution model.  The original documentation of this
   * method is below.
   * \sa ProcessObject::GenerateOutputInformaton() */
  void
  GenerateOutputInformation() override;

  /** DownsampleImageFilter needs a larger input requested region than the output
   * requested region.  As such, DownsampleImageFilter needs to provide an
   * implementation for GenerateInputRequestedRegion() in order to inform the
   * pipeline execution model.
   * \sa ProcessObject::GenerateInputRequestedRegion() */
  void
  GenerateInputRequestedRegion() override;

protected:
  DownsampleImageFilter();
  ~DownsampleImageFilter() override = default;

  /** DownsampleImageFilter can be implemented as a multithreaded filter.
   * Therefore, this implementation provides a ThreadedGenerateData() routine
   * which is called for each processing thread. The output image data is
   * allocated automatically by the superclass prior to calling
   * ThreadedGenerateData().  ThreadedGenerateData can only write to the
   * portion of the output image specified by the parameter
   * "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData() */
  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  unsigned int m_Factors[ImageDimension];
  int          m_Offsets[ImageDimension];
};


} // end namespace rtk

#ifndef rtk_MANUAL_INSTANTIATION
#  include "rtkDownsampleImageFilter.hxx"
#endif

#endif
