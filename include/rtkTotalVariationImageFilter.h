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

#ifndef rtkTotalVariationImageFilter_h
#define rtkTotalVariationImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkNumericTraits.h>
#include <itkArray.h>
#include <itkSimpleDataObjectDecorator.h>

#include "rtkMacro.h"

namespace rtk
{
/** \class TotalVariationImageFilter
 * \brief Compute the total variation of an Image.
 *
 * TotalVariationImageFilter computes the total variation, defined
 * as the L1 norm of the image of the L2 norm of the gradient,
 * of an image. The filter needs all of its input image.  It
 * behaves as a filter with an input and output. Thus it can be inserted
 * in a pipeline with other filters and the total variation will only be
 * recomputed if a downstream filter changes.
 *
 * The filter passes its input through unmodified. The filter is
 * threaded.
 *
 * \ingroup RTK MathematicalStatisticsImageFilters
 * \ingroup RTK ITKImageStatistics
 *
 */

template <typename TInputImage>
class ITK_TEMPLATE_EXPORT TotalVariationImageFilter : public itk::ImageToImageFilter<TInputImage, TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TotalVariationImageFilter);

  /** Standard Self type alias */
  using Self = TotalVariationImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TInputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(TotalVariationImageFilter);

  /** Image related type alias. */
  using InputImagePointer = typename TInputImage::Pointer;

  using RegionType = typename TInputImage::RegionType;
  using SizeType = typename TInputImage::SizeType;
  using IndexType = typename TInputImage::IndexType;
  using PixelType = typename TInputImage::PixelType;

  /** Image related type alias. */
  static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;

  /** Type to use for computations. */
  using RealType = typename itk::NumericTraits<PixelType>::RealType;

  /** Smart Pointer type to a DataObject. */
  using DataObjectPointer = typename itk::DataObject::Pointer;

  /** Type of DataObjects used for scalar outputs */
  using RealObjectType = itk::SimpleDataObjectDecorator<RealType>;
  //  using PixelObjectType = SimpleDataObjectDecorator< PixelType >;

  /** Return the computed Minimum. */
  RealType
  GetTotalVariation() const
  {
    return this->GetTotalVariationOutput()->Get();
  }
  RealObjectType *
  GetTotalVariationOutput();

  const RealObjectType *
  GetTotalVariationOutput() const;

  /** Make a DataObject of the correct type to be used as the specified
   * output. */
  using DataObjectPointerArraySizeType = itk::ProcessObject::DataObjectPointerArraySizeType;
  using Superclass::MakeOutput;
  DataObjectPointer
  MakeOutput(DataObjectPointerArraySizeType output) override;

  // Begin concept checking
  itkConceptMacro(InputHasNumericTraitsCheck, (itk::Concept::HasNumericTraits<PixelType>));
  // End concept checking

  /** Use the image spacing information in calculations. Use this option if you
   *  want derivatives in physical space. Default is UseImageSpacingOn. */
  void
  SetUseImageSpacingOn()
  {
    this->SetUseImageSpacing(true);
  }

  /** Ignore the image spacing. Use this option if you want derivatives in
      isotropic pixel space.  Default is UseImageSpacingOn. */
  void
  SetUseImageSpacingOff()
  {
    this->SetUseImageSpacing(false);
  }

  /** Set/Get whether or not the filter will use the spacing of the input
      image in its calculations */
  itkSetMacro(UseImageSpacing, bool);
  itkGetConstMacro(UseImageSpacing, bool);

protected:
  TotalVariationImageFilter();
  ~TotalVariationImageFilter() override = default;
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

  /** Pass the input through unmodified. Do this by Grafting in the
   *  AllocateOutputs method.
   */
  void
  AllocateOutputs() override;

  /** Initialize some accumulators before the threads run. */
  void
  BeforeThreadedGenerateData() override;

  /** Do final mean and variance computation from data accumulated in threads.
   */
  void
  AfterThreadedGenerateData() override;

  /** Multi-thread version GenerateData. */
  void
  ThreadedGenerateData(const RegionType & outputRegionForThread, itk::ThreadIdType threadId) override;

  // Override since the filter needs all the data for the algorithm
  void
  GenerateInputRequestedRegion() override;

  // Override since the filter produces all of its output
  void
  EnlargeOutputRequestedRegion(itk::DataObject * data) override;

  bool m_UseImageSpacing;

private:
  itk::Array<RealType> m_SumOfSquareRoots;
}; // end of class
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkTotalVariationImageFilter.hxx"
#endif

#endif
