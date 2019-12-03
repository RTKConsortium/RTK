/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
#ifndef itkCudaImageToImageFilter_h
#define itkCudaImageToImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkCudaKernelManager.h"

namespace itk
{

/** \class CudaImageToImageFilter
 *
 * \brief class to abstract the behaviour of the Cuda filters.
 *
 * CudaImageToImageFilter is the Cuda version of ImageToImageFilter.
 * This class can accept both CPU and GPU image as input and output,
 * and apply filter accordingly. If Cuda is available for use, then
 * GPUGenerateData() is called. Otherwise, GenerateData() in the
 * parent class (i.e., ImageToImageFilter) will be called.
 *
 * \ingroup ITKCudaCommon
 */
template <class TInputImage,
          class TOutputImage,
          class TParentImageFilter = ImageToImageFilter<TInputImage, TOutputImage>>
class ITK_EXPORT CudaImageToImageFilter : public TParentImageFilter
{
public:
  /** Standard class type alias. */
  using Self = CudaImageToImageFilter;
  using Superclass = TParentImageFilter;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaImageToImageFilter, TParentImageFilter);

  /** Superclass type alias. */
  using DataObjectIdentifierType = typename Superclass::DataObjectIdentifierType;
  using OutputImageRegionType = typename Superclass::OutputImageRegionType;
  using OutputImagePixelType = typename Superclass::OutputImagePixelType;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using InputImagePointer = typename InputImageType::Pointer;
  using InputImageConstPointer = typename InputImageType::ConstPointer;
  using InputImageRegionType = typename InputImageType::RegionType;
  using InputImagePixelType = typename InputImageType::PixelType;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;
  static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;

  // macro to set if Cuda is used
  itkSetMacro(GPUEnabled, bool);
  itkGetConstMacro(GPUEnabled, bool);
  itkBooleanMacro(GPUEnabled);

  void
  GenerateData() override;
  virtual void
  GraftOutput(typename itk::CudaTraits<TOutputImage>::Type * output);
  virtual void
  GraftOutput(const DataObjectIdentifierType & key, typename itk::CudaTraits<TOutputImage>::Type * output);

protected:
  void
  GraftOutput(DataObject * output) override;
  void
  GraftOutput(const DataObjectIdentifierType & key, DataObject * output) override;
  CudaImageToImageFilter();
  ~CudaImageToImageFilter();

  virtual void
  PrintSelf(std::ostream & os, Indent indent) const;

  virtual void
  GPUGenerateData()
  {}

  // Cuda kernel manager
  typename CudaKernelManager::Pointer m_CudaKernelManager;

private:
  CudaImageToImageFilter(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented

  bool m_GPUEnabled;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaImageToImageFilter.hxx"
#endif

#endif
