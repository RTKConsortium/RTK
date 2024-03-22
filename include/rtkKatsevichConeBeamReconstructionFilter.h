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

#ifndef rtkKatsevichConeBeamReconstructionFilter_h
#define rtkKatsevichConeBeamReconstructionFilter_h

#include <itkExtractImageFilter.h>
#include "rtkKatsevichDerivativeImageFilter.h"
#include "rtkKatsevichForwardBinningImageFilter.h"
#include "rtkFFTHilbertImageFilter.h"
#include "rtkKatsevichBackwardBinningImageFilter.h"
#include "rtkKatsevichBackProjectionImageFilter.h"

#include "rtkConfiguration.h"


namespace rtk
{

/** \class KatsevichConeBeamReconstructionFilter
 * \brief Implements Feldkamp, David and Kress cone-beam reconstruction
 *
 * KatsevichConeBeamReconstructionFilter is a mini-pipeline filter which combines
 * the different steps of the Katsevich cone-beam reconstruction filter:
 * - rtk::KatsevichDerivativeImageFilter for derivative and cosine-weighting of the projections,
 * - rtk::KatsevichForwardBinningImageFilter, rtk::FFTHilbertImageFilter an d rtk::KatsevichBackwardBinningImageFilter
 * for Hilbert filtering,
 * - rtk::KatsevichBackProjectionImageFilter for backprojection.
 * The input stack of projections is processed piece by piece (the size is
 * controlled with ProjectionSubsetSize) via the use of itk::ExtractImageFilter
 * to extract sub-stacks.
 * Note though that this behaviour only mimicks the FDK filter. The ProjectionSubsetSize is set
 * to the total number of projections (because the derivative filter does not process projections
 * independently)
 *
 * \dot
 * digraph KatsevichConeBeamReconstructionFilter {
 * node [shape=box];
 * 1 [ label="rtk::KatsevichDerivativeImageFilter" URL="\ref rtk::KatsevichDerivativeImageFilter"];
 * 2 [ label="rtk::KatsevichForwardBinningImageFilter" URL="\ref rtk::KatsevichForwardBinningImageFilter"];
 * 3 [ label="rtk::FFTHilbertImageFilter" URL="\ref rtk::FFTHilbertImageFilter"];
 * 4 [ label="rtk::KatsevichBackwardBinningImageFilter" URL="\ref rtk::KatsevichBackwardBinningImageFilter"];
 * 5 [ label="rtk::KatsevichBackProjectionImageFilter" URL="\ref rtk::KatsevichBackProjectionImageFilter"];
 * 1 -> 2;
 * 2 -> 3;
 * 3 -> 4;
 * 4 -> 5;
 * }
 * \enddot
 *
 * \test
 *
 * \author Jerome Lesaint
 *
 * \ingroup RTK ReconstructionAlgorithm
 */
template <class TInputImage, class TOutputImage = TInputImage, class TFFTPrecision = double>
class ITK_TEMPLATE_EXPORT KatsevichConeBeamReconstructionFilter
  : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(KatsevichConeBeamReconstructionFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(KatsevichConeBeamReconstructionFilter);
#endif

  /** Standard class type alias. */
  using Self = KatsevichConeBeamReconstructionFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;

  /** Typedefs of each subfilter of this composite filter */
  using ExtractFilterType = itk::ExtractImageFilter<InputImageType, OutputImageType>;
  using DerivativeFilterType = rtk::KatsevichDerivativeImageFilter<InputImageType, OutputImageType>;
  using ForwardFilterType = rtk::KatsevichForwardBinningImageFilter<OutputImageType, OutputImageType>;
  using HilbertFilterType = rtk::FFTHilbertImageFilter<OutputImageType, OutputImageType, TFFTPrecision>;
  using BackwardFilterType = rtk::KatsevichBackwardBinningImageFilter<InputImageType, OutputImageType>;
  using BackProjectionFilterType = rtk::KatsevichBackProjectionImageFilter<OutputImageType, OutputImageType>;
  using BackProjectionFilterPointer = typename BackProjectionFilterType::Pointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(KatsevichConeBeamReconstructionFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetModifiableObjectMacro(Geometry, ThreeDHelicalProjectionGeometry);
  itkSetObjectMacro(Geometry, ThreeDHelicalProjectionGeometry);

  /** Get pointer to the extract filter used by the Katsevich reconstruction */
  typename DerivativeFilterType::Pointer
  GetExtractFilter()
  {
    return m_ExtractFilter;
  }

  /** Get pointer to the derivative filter used by the Katsevich reconstruction */
  typename DerivativeFilterType::Pointer
  GetDerivativeFilter()
  {
    return m_DerivativeFilter;
  }

  /** Get pointer to the forward binning filter used by the Katsevich reconstruction */
  typename ForwardFilterType::Pointer
  GetForwardFilter()
  {
    return m_ForwardFilter;
  }

  /** Get pointer to the Hilbert filter used by the Katsevich reconstruction */
  typename HilbertFilterType::Pointer
  GetHilbertFilter()
  {
    return m_HilbertFilter;
  }

  /** Get pointer to the forward binning filter used by the Katsevich reconstruction */
  typename BackwardFilterType::Pointer
  GetBackwardFilter()
  {
    return m_BackwardFilter;
  }

  /** Get pointer to the backprojection filter used by the Katsevich reconstruction */
  typename BackProjectionFilterType::Pointer
  GetBackProjectionFilter()
  {
    return m_BackProjectionFilter;
  }

  ////////////////////////////
  // JL : the default here is set to the total number of projections.
  // In another words, the extract filter does NOTHING !
  ////////////////////////////
  /** Get / Set the number of cone-beam projection images processed
      simultaneously. Default is 4. */
  itkGetMacro(ProjectionSubsetSize, unsigned int);
  itkSetMacro(ProjectionSubsetSize, unsigned int);

  ///** Get / Set and init the backprojection filter. The set function takes care
  // * of initializing the mini-pipeline and the ramp filter must therefore be
  // * created before calling this set function. */
  // itkGetMacro(BackProjectionFilter, BackProjectionFilterPointer);
  // virtual void
  // SetBackProjectionFilter(const BackProjectionFilterPointer _arg);

protected:
  KatsevichConeBeamReconstructionFilter();
  ~KatsevichConeBeamReconstructionFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() ITKv5_CONST override;

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
  typename ExtractFilterType::Pointer    m_ExtractFilter;
  typename DerivativeFilterType::Pointer m_DerivativeFilter;
  typename ForwardFilterType::Pointer    m_ForwardFilter;
  typename HilbertFilterType::Pointer    m_HilbertFilter;
  typename BackwardFilterType::Pointer   m_BackwardFilter;
  BackProjectionFilterPointer            m_BackProjectionFilter;

private:
  /** Number of projections processed at a time. */
  unsigned int m_ProjectionSubsetSize{ 16 };

  /** Geometry propagated to subfilters of the mini-pipeline. */
  ThreeDHelicalProjectionGeometry::Pointer m_Geometry;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkKatsevichConeBeamReconstructionFilter.hxx"
#endif

#endif
