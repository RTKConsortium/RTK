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

#ifndef rtkSoftThresholdTVImageFilter_h
#define rtkSoftThresholdTVImageFilter_h

#include <itkNeighborhoodIterator.h>
#include <itkImageToImageFilter.h>
#include <itkImage.h>
#include <itkVector.h>
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector_fixed.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "rtkConfiguration.h"

namespace rtk
{
/** \class SoftThresholdTVImageFilter
 *
 * \brief Computes the Total Variation from a gradient input image
 * (pixels are vectors), soft thresholds it, and outputs a
 *  multiple channel image with vectors colinear to the input vectors
 *  but having a smaller norm.
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <typename TInputImage, typename TRealType = float, typename TOutputImage = TInputImage>
class ITK_EXPORT SoftThresholdTVImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(SoftThresholdTVImageFilter);

  /** Standard class type alias. */
  using Self = SoftThresholdTVImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(SoftThresholdTVImageFilter, ImageToImageFilter);

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
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Length of the vector pixel type of the input image. */
  itkStaticConstMacro(VectorDimension, unsigned int, InputPixelType::Dimension);

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

  itkGetMacro(Threshold, float);
  itkSetMacro(Threshold, float);

protected:
  SoftThresholdTVImageFilter();
  ~SoftThresholdTVImageFilter() override = default;

  /** Do any necessary casting/copying of the input data.  Input pixel types
     whose value types are not real number types must be cast to real number
     types. */
  //    void BeforeThreadedGenerateData();

  /** SoftThresholdTVImageFilter can be implemented as a
   * multithreaded filter.  Therefore, this implementation provides a
   * ThreadedGenerateData() routine which is called for each
   * processing thread. The output image data is allocated
   * automatically by the superclass prior to calling
   * ThreadedGenerateData().  ThreadedGenerateData can only write to
   * the portion of the output image specified by the parameter
   * "outputRegionForThread"
   *
   * \sa ImageToImageFilter::ThreadedGenerateData(),
   *     ImageToImageFilter::GenerateData() */
  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  //    using ImageBaseType = typename InputImageType::Superclass;

private:
  float        m_Threshold;
  ThreadIdType m_RequestedNumberOfThreads;
};
} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSoftThresholdTVImageFilter.hxx"
#endif

#endif
