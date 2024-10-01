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

#ifndef rtkKatsevichDerivativeImageFilter_h
#define rtkKatsevichDerivativeImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkNumericTraits.h"
#include "itkIndent.h"

#include "itkConstShapedNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"

#include "itkImageSliceIteratorWithIndex.h"

#include "rtkThreeDHelicalProjectionGeometry.h"


namespace rtk
{

/** \class KatsevichDerivativeImageFilter
 * \brief Computes the derivatives of the projections wrt the projection index (the rotation angle).
 *
 * This filter implements the derivative formula Eq. 46 (curved det) and Eq. 87 (flat panel)
 * from Noo et al., PMB, 2003
 *
 *
 * \author Jerome Lesaint
 *
 * \ingroup
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT KatsevichDerivativeImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(KatsevichDerivativeImageFilter);

  /** Standard class type alias. */
  using Self = KatsevichDerivativeImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(KatsevichDerivativeImageFilter, ImageToImageFilter);

  /** Typedef to images */
  using OutputImageType = TOutputImage;
  using InputImageType = TInputImage;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using InputImagePointer = typename InputImageType::Pointer;
  using InputImageConstPointer = typename InputImageType::ConstPointer;
  using OutputImageRegionType = typename TOutputImage::RegionType;

  using ConstShapedNeighborhoodIteratorType = itk::ConstShapedNeighborhoodIterator<InputImageType>;
  using IteratorType = itk::ImageRegionIterator<InputImageType>;
  using SliceIteratorType = itk::ImageSliceIteratorWithIndex<InputImageType>;
  using IndexType = typename InputImageType::IndexType;


  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputInternalPixelType = typename TOutputImage::InternalPixelType;
  using InputPixelType = typename TInputImage::PixelType;
  using InputInternalPixelType = typename TInputImage::InternalPixelType;
  static constexpr unsigned int ImageDimension = TOutputImage::ImageDimension;

  /** Get/ Set geometry structure */
  itkGetMacro(Geometry, ::rtk::ThreeDHelicalProjectionGeometry::Pointer);
  itkSetObjectMacro(Geometry, ::rtk::ThreeDHelicalProjectionGeometry);


  void
  GenerateOutputInformation() override;

  void
  GenerateInputRequestedRegion() override;

protected:
  KatsevichDerivativeImageFilter();
  ~KatsevichDerivativeImageFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() ITKv5_CONST override;

  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

  /** Standard pipeline method. While this class does not implement a
   * ThreadedGenerateData(), its GenerateData() delegates all
   * calculations to an NeighborhoodOperatorImageFilter.  Since the
   * NeighborhoodOperatorImageFilter is multithreaded, this filter is
   * multithreaded by default. */
  void
  GenerateData() override;

  // void
  // DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  ThreeDHelicalProjectionGeometry::Pointer m_Geometry;
};


} // end namespace rtk

#ifndef rtk_MANUAL_INSTANTIATION
#  include "rtkKatsevichDerivativeImageFilter.hxx"
#endif

#endif
