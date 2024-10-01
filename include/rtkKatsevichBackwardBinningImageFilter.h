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

#ifndef rtkKatsevichBackwardBinningImageFilter_h
#define rtkKatsevichBackwardBinningImageFilter_h

#include <rtkConfiguration.h>

#include <itkInPlaceImageFilter.h>
#include <itkConceptChecking.h>

#include <rtkThreeDHelicalProjectionGeometry.h>
#include <rtkConstantImageSource.h>
#include "itkImageSliceIteratorWithIndex.h"

#include <type_traits>
#include <typeinfo>

namespace rtk
{

/** \class KatsevichBackwardBinningImageFilter
 * \brief
 *
 * Resample the Hilbert transformed data back to the original v-coordinate, according
 * to a specified geometry.
 *
 * \test
 *
 * \author Jerome Lesaint
 *
 * \ingroup RTK Projector
 */
template <class TInputImage, class TOutputImage>
class KatsevichBackwardBinningImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(KatsevichBackwardBinningImageFilter);

  /** Standard class type alias. */
  using Self = KatsevichBackwardBinningImageFilter;
  using OutputImageType = TOutputImage;
  using InputImageType = TInputImage;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using InternalInputPixelType = typename TInputImage::InternalPixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using OutputImageIndexType = typename TOutputImage::IndexType;
  using OutputImageSizeType = typename TOutputImage::SizeType;
  using InputImageRegionType = typename TInputImage::RegionType;
  using InputImageIndexType = typename TInputImage::IndexType;
  using InputImageSizeType = typename TInputImage::SizeType;
  using InputSliceIteratorType = itk::ImageSliceIteratorWithIndex<const InputImageType>;
  using OutputSliceIteratorType = itk::ImageSliceIteratorWithIndex<OutputImageType>;
  using ConstantImageType = rtk::ConstantImageSource<OutputImageType>;


  using GeometryType = rtk::ThreeDHelicalProjectionGeometry;
  using GeometryPointer = GeometryType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(KatsevichBackwardBinningImageFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Get / Set the transpose flag for 2D projections (optimization trick) */
  itkGetMacro(Transpose, bool);
  itkSetMacro(Transpose, bool);

protected:
  KatsevichBackwardBinningImageFilter() = default;
  ~KatsevichBackwardBinningImageFilter() override = default;

  void
  GenerateOutputInformation() override;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() ITKv5_CONST override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

  /** The input is a stack of projections, we need to interpolate in one projection
      for efficiency during interpolation. Use of itk::ExtractImageFilter is
      not threadsafe in ThreadedGenerateData, this one is. The output can be multiplied by a constant.
      The function is templated to allow getting an itk::CudaImage. */
  template <class TProjectionImage>
  typename TProjectionImage::Pointer
  GetProjection(const unsigned int iProj);

  /** RTK geometry object */
  GeometryPointer m_Geometry;

private:
  /** Flip projection flag: infludences GetProjection and
    GetIndexToIndexProjectionMatrix for optimization */
  bool m_Transpose{ false };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkKatsevichBackwardBinningImageFilter.hxx"
#endif

#endif
