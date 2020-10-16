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

#ifndef rtkUpsampleImageFilter_h
#define rtkUpsampleImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkImageRegionSplitterDirection.h>

namespace rtk
{

/** \class UpsampleImageFilter
 * \brief Upsamples an image by the given factor for each dimension.
 *
 * This filter is inspired from Dan Mueller's GIFT package
 * http://www.insight-journal.org/browse/publication/103
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT UpsampleImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(UpsampleImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(UpsampleImageFilter);
#endif

  /** Standard class type alias. */
  using Self = UpsampleImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(UpsampleImageFilter, ImageToImageFilter);

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

  /** Set the shrink factors. Values are clamped to
   * a minimum value of 1.*/
  void
  SetFactors(const unsigned int factors[]);

  /** Sets the shrink factor for the given dimension.
   * All other dimensions are set to 1 */
  void
  SetFactor(unsigned int dimension, unsigned int factor);

  /** UpsampleImageFilter produces an image which is a different
   * resolution and with a different pixel spacing than its input
   * image.  As such, UpsampleImageFilter needs to provide an
   * implementation for GenerateOutputInformation() in order to inform
   * the pipeline execution model.  The original documentation of this
   * method is below.
   * \sa ProcessObject::GenerateOutputInformaton() */
  void
  GenerateOutputInformation() override;

  /** UpsampleImageFilter needs a larger input requested region than the output
   * requested region.  As such, UpsampleImageFilter needs to provide an
   * implementation for GenerateInputRequestedRegion() in order to inform the
   * pipeline execution model.
   * \sa ProcessObject::GenerateInputRequestedRegion() */
  void
  GenerateInputRequestedRegion() override;

  /** Set/Get the order of the wavelet filter
   * This is required because some information about the index of the image
   * is lost during downsampling, and the upsampling filter can't guess
   * what the exact index should be.
   */
  itkSetMacro(Order, unsigned int);
  itkGetMacro(Order, unsigned int);

  /** Set/Get the size of the output image
   * This is required because some information about the size of the image
   * is lost during downsampling, and the upsampling filter can't guess
   * what the exact size should be.
   */
  itkSetMacro(OutputSize, typename TOutputImage::SizeType);
  itkGetMacro(OutputSize, typename TOutputImage::SizeType);

  /** Set/Get the index of the output image
   * This is required because some information about the index of the image
   * is lost during downsampling, and the upsampling filter can't guess
   * what the exact index should be. The output index is actually set to
   * OutputIndex + 1.
   */
  itkSetMacro(OutputIndex, typename TOutputImage::IndexType);
  itkGetMacro(OutputIndex, typename TOutputImage::IndexType);

protected:
  UpsampleImageFilter();
  ~UpsampleImageFilter() override = default;

  /** UpsampleImageFilter can be implemented as a multithreaded filter.
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
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                       itk::ThreadIdType             itkNotUsed(threadId)) override;

private:
  unsigned int                     m_Factors[ImageDimension];
  unsigned int                     m_Order;
  typename TOutputImage::SizeType  m_OutputSize;
  typename TOutputImage::IndexType m_OutputIndex;

  const itk::ImageRegionSplitterBase *
                                             GetImageRegionSplitter() const override;
  itk::ImageRegionSplitterDirection::Pointer m_Splitter;
};


} // end namespace rtk

#ifndef rtk_MANUAL_INSTANTIATION
#  include "rtkUpsampleImageFilter.hxx"
#endif

#endif
