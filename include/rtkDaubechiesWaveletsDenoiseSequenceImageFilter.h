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

#ifndef rtkDaubechiesWaveletsDenoiseSequenceImageFilter_h
#define rtkDaubechiesWaveletsDenoiseSequenceImageFilter_h

#include "rtkConstantImageSource.h"

#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkCastImageFilter.h>
#include "rtkDeconstructSoftThresholdReconstructImageFilter.h"

namespace rtk
{
/** \class DaubechiesWaveletsDenoiseSequenceImageFilter
 * \brief Applies 3D Daubechies wavelets denoising to a 3D + time sequence of images
 *
 * Most of the work in this filter is performed by the underlying
 * rtkDeconstructSoftThresholdReconstructImageFilter
 *
 * \dot
 * digraph DaubechiesWaveletsDenoiseSequenceImageFilter {
 *
 * Input0 [ label="Input 0 (Sequence of images)"];
 * Input0 [shape=Mdiamond];
 * Output [label="Output (Sequence of images)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * Extract [label="itk::ExtractImageFilter (for images)" URL="\ref itk::ExtractImageFilter"];
 * WaveletsDenoising [ label="rtk::DeconstructSoftThresholdReconstructImageFilter"
 *                     URL="\ref rtk::DeconstructSoftThresholdReconstructImageFilter"];
 * Cast [ label="itk::CastImageFilter" URL="\ref itk::CastImageFilter"];
 * Paste [ label="itk::PasteImageFilter" URL="\ref itk::PasteImageFilter"];
 * ConstantSource [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * BeforePaste [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterPaste [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 * Input0 -> Extract;
 * Extract -> WaveletsDenoising;
 * WaveletsDenoising -> Cast;
 * Cast -> BeforePaste [arrowhead=none];
 * BeforePaste -> Paste;
 * Paste -> AfterPaste [arrowhead=none];
 * AfterPaste -> BeforePaste [style=dashed];
 * AfterPaste -> Output [style=dashed];
 * }
 * \enddot
 *
 * \test rtkfourdroostertest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename TImageSequence>
class DaubechiesWaveletsDenoiseSequenceImageFilter : public itk::ImageToImageFilter<TImageSequence, TImageSequence>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(DaubechiesWaveletsDenoiseSequenceImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(DaubechiesWaveletsDenoiseSequenceImageFilter);
#endif

  /** Standard class type alias. */
  using Self = DaubechiesWaveletsDenoiseSequenceImageFilter;
  using Superclass = itk::ImageToImageFilter<TImageSequence, TImageSequence>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DaubechiesWaveletsDenoiseSequenceImageFilter, ImageToImageFilter);

  /** Set the number of levels of the deconstruction and reconstruction */
  itkGetMacro(NumberOfLevels, unsigned int);
  itkSetMacro(NumberOfLevels, unsigned int);

  /** Sets the order of the Daubechies wavelet used to deconstruct/reconstruct the image pyramid */
  itkGetMacro(Order, unsigned int);
  itkSetMacro(Order, unsigned int);

  /** Sets the threshold used in soft thresholding */
  itkGetMacro(Threshold, float);
  itkSetMacro(Threshold, float);

  /** Typedefs of internal filters */
  using TImage = itk::Image<typename TImageSequence::PixelType, TImageSequence::ImageDimension - 1>;
  using WaveletsDenoisingFilterType = rtk::DeconstructSoftThresholdReconstructImageFilter<TImage>;
  using ExtractFilterType = itk::ExtractImageFilter<TImageSequence, TImage>;
  using PasteFilterType = itk::PasteImageFilter<TImageSequence, TImageSequence>;
  using CastFilterType = itk::CastImageFilter<TImage, TImageSequence>;
  using ConstantImageSourceType = rtk::ConstantImageSource<TImageSequence>;

protected:
  DaubechiesWaveletsDenoiseSequenceImageFilter();
  ~DaubechiesWaveletsDenoiseSequenceImageFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  void
  GenerateOutputInformation() override;
  void
  GenerateInputRequestedRegion() override;

  /** Member pointers to the filters used internally (for convenience)*/
  typename WaveletsDenoisingFilterType::Pointer m_WaveletsDenoisingFilter;
  typename ExtractFilterType::Pointer           m_ExtractFilter;
  typename PasteFilterType::Pointer             m_PasteFilter;
  typename CastFilterType::Pointer              m_CastFilter;
  typename ConstantImageSourceType::Pointer     m_ConstantSource;

  /** Extraction regions for both extract filters */
  typename TImageSequence::RegionType m_ExtractAndPasteRegion;

  /** Information for the wavelets denoising filter */
  unsigned int m_Order{ 5 };
  float        m_Threshold{ 1 };
  unsigned int m_NumberOfLevels{ 3 };
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDaubechiesWaveletsDenoiseSequenceImageFilter.hxx"
#endif

#endif
