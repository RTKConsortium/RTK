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

#ifndef rtkMagnitudeThresholdImageFilter_h
#define rtkMagnitudeThresholdImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkVector.h>

namespace rtk
{
/** \class MagnitudeThresholdImageFilter
 *
 * \brief Performs thresholding on the norm of each vector-valued input pixel
 *
 * If the norm of a vector is higher than the threshold, divides the
 * components of the vector by norm / threshold. Mathematically, it amounts
 * to projecting onto the L_2 ball of radius m_Threshold
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <typename TInputImage, typename TRealType = float, typename TOutputImage = TInputImage>
class ITK_TEMPLATE_EXPORT MagnitudeThresholdImageFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MagnitudeThresholdImageFilter);

  /** Standard class type alias. */
  using Self = MagnitudeThresholdImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(MagnitudeThresholdImageFilter);
#else
  itkTypeMacro(MagnitudeThresholdImageFilter, ImageToImageFilter);
#endif

  /** Extract some information from the image types.  Dimensionality
   * of the two images is assumed to be the same. */
  using OutputPixelType = typename TOutputImage::PixelType;
  using InputPixelType = typename TInputImage::PixelType;

  /** Image type alias support */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using InputImagePointer = typename InputImageType::Pointer;
  using OutputImagePointer = typename OutputImageType::Pointer;

  /** The dimensionality of the input and output images. */
  static constexpr unsigned int ImageDimension = TOutputImage::ImageDimension;

  /** Length of the vector pixel type of the input image. */
  static constexpr unsigned int VectorDimension = InputPixelType::Dimension;

  /** Define the data type and the vector of data type used in calculations. */
  using RealType = TRealType;
  using RealVectorType = itk::Vector<TRealType, InputPixelType::Dimension>;
  using RealVectorImageType = itk::Image<RealVectorType, TInputImage::ImageDimension>;

  /** Superclass type alias. */
  using OutputImageRegionType = typename Superclass::OutputImageRegionType;

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasNumericTraitsCheck, (itk::Concept::HasNumericTraits<typename InputPixelType::ValueType>));
  itkConceptMacro(RealTypeHasNumericTraitsCheck, (itk::Concept::HasNumericTraits<RealType>));
  /** End concept checking */
#endif

  itkGetMacro(Threshold, TRealType);
  itkSetMacro(Threshold, TRealType);

protected:
  MagnitudeThresholdImageFilter();
  ~MagnitudeThresholdImageFilter() override = default;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  TRealType m_Threshold;
};
} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkMagnitudeThresholdImageFilter.hxx"
#endif

#endif
