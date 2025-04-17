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

#ifndef rtkFDKVarianceReconstructionFilter_h
#define rtkFDKVarianceReconstructionFilter_h

#include "rtkFDKWeightProjectionFilter.h"
#include "rtkFFTVarianceRampImageFilter.h"
#include "rtkFDKBackProjectionImageFilter.h"
#include "rtkConfiguration.h"

#include <itkExtractImageFilter.h>

namespace rtk
{

/** \class FDKVarianceReconstructionFilter
 * \brief Implements reconstruction of the variance map of images reconstructed with FDK
 *
 * \dot
 * digraph FDKVarianceReconstructionFilter {
 *  node [shape=box];
 *  1 [ label="rtk::FDKWeightProjectionFilter" URL="\ref rtk::FDKWeightProjectionFilter"];
 *  2 [ label="rtk::FDKWeightProjectionFilter" URL="\ref rtk::FDKWeightProjectionFilter"];
 *  3 [ label="rtk::FFTVarianceRampImageFilter" URL="\ref rtk::FFTVarianceRampImageFilter"];
 *  4 [ label="rtk::FDKBackProjectionImageFilter" URL="\ref rtk::FDKBackProjectionImageFilter"];
 *  1 -> 2;
 *  2 -> 3;
 *  3 -> 4;
 * }
 * \enddot
 *
 * \test rtkvariancereconstructiontest
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
template <class TInputImage, class TOutputImage = TInputImage, class TFFTPrecision = double>
class ITK_TEMPLATE_EXPORT FDKVarianceReconstructionFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FDKVarianceReconstructionFilter);

  /** Standard class type alias. */
  using Self = FDKVarianceReconstructionFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;

  /** Typedefs of each subfilter of this composite filter */
  using ExtractFilterType = itk::ExtractImageFilter<InputImageType, OutputImageType>;
  using WeightFilterType = rtk::FDKWeightProjectionFilter<InputImageType, OutputImageType>;
  using VarianceRampFilterType = rtk::FFTVarianceRampImageFilter<OutputImageType, OutputImageType, TFFTPrecision>;
  using BackProjectionFilterType = rtk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType>;
  using BackProjectionFilterPointer = typename BackProjectionFilterType::Pointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FDKVarianceReconstructionFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetModifiableObjectMacro(Geometry, ThreeDCircularProjectionGeometry);
  itkSetObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

  /** Get pointer to the ramp filter used by the feldkamp reconstruction */
  typename VarianceRampFilterType::Pointer
  GetVarianceRampFilter()
  {
    return m_VarianceRampFilter;
  }

  /** Get / Set the number of cone-beam projection images processed
      simultaneously. Default is 4. */
  itkGetMacro(ProjectionSubsetSize, unsigned int);
  itkSetMacro(ProjectionSubsetSize, unsigned int);

  /** Get / Set and init the backprojection filter. The set function takes care
   * of initializing the mini-pipeline and the ramp filter must therefore be
   * created before calling this set function. */
  itkGetMacro(BackProjectionFilter, BackProjectionFilterPointer);
  virtual void
  SetBackProjectionFilter(const BackProjectionFilterPointer _arg);

protected:
  FDKVarianceReconstructionFilter();
  ~FDKVarianceReconstructionFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() const override;

  void
  GenerateInputRequestedRegion() override;

  void
  GenerateOutputInformation() override;

  void
  GenerateData() override;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

  /** Pointers to each subfilter of this composite filter */
  typename ExtractFilterType::Pointer      m_ExtractFilter;
  typename WeightFilterType::Pointer       m_WeightFilter1;
  typename WeightFilterType::Pointer       m_WeightFilter2;
  typename VarianceRampFilterType::Pointer m_VarianceRampFilter;
  BackProjectionFilterPointer              m_BackProjectionFilter;

private:
  /** Number of projections processed at a time. */
  unsigned int m_ProjectionSubsetSize{ 16 };

  /** Geometry propagated to subfilters of the mini-pipeline. */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFDKVarianceReconstructionFilter.hxx"
#endif

#endif
