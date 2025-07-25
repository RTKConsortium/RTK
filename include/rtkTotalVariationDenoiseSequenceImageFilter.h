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

#ifndef rtkTotalVariationDenoiseSequenceImageFilter_h
#define rtkTotalVariationDenoiseSequenceImageFilter_h

#include "rtkConstantImageSource.h"

#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkCastImageFilter.h>

#ifdef RTK_USE_CUDA
#  include "rtkCudaTotalVariationDenoisingBPDQImageFilter.h"
#else
#  include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#endif

namespace rtk
{
/** \class TotalVariationDenoiseSequenceImageFilter
 * \brief Applies 3D total variation denoising to a 3D + time sequence of images
 *
 * Most of the work in this filter is performed by the underlying rtkTotalVariationDenoisingBPDQImageFilter
 * or its CUDA version
 *
 * \dot
 * digraph TotalVariationDenoiseSequenceImageFilter {
 *
 * Input0 [ label="Input 0 (Sequence of images)"];
 * Input0 [shape=Mdiamond];
 * Output [label="Output (Sequence of images)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * Extract [label="itk::ExtractImageFilter (for images)" URL="\ref itk::ExtractImageFilter"];
 * TVDenoising [ label="rtk::TotalVariationDenoisingBPDQImageFilter"
 *               URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
 * Cast [ label="itk::CastImageFilter" URL="\ref itk::CastImageFilter"];
 * Paste [ label="itk::PasteImageFilter" URL="\ref itk::PasteImageFilter"];
 * ConstantSource [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * BeforePaste [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterPaste [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 * Input0 -> Extract;
 * Extract -> TVDenoising;
 * TVDenoising -> Cast;
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
class ITK_TEMPLATE_EXPORT TotalVariationDenoiseSequenceImageFilter
  : public itk::ImageToImageFilter<TImageSequence, TImageSequence>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TotalVariationDenoiseSequenceImageFilter);

  /** Standard class type alias. */
  using Self = TotalVariationDenoiseSequenceImageFilter;
  using Superclass = itk::ImageToImageFilter<TImageSequence, TImageSequence>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(TotalVariationDenoiseSequenceImageFilter);

  /** Set/Get for the TotalVariationDenoisingBPDQImageFilter */
  itkGetMacro(Gamma, double);
  itkSetMacro(Gamma, double);

  itkGetMacro(NumberOfIterations, int);
  itkSetMacro(NumberOfIterations, int);

  void
  SetDimensionsProcessed(bool * arg);

  /** SFINAE type alias, depending on whether a CUDA image is used. */
  using CPUImageSequenceType = typename itk::Image<typename TImageSequence::PixelType, TImageSequence::ImageDimension>;
#ifdef RTK_USE_CUDA
  using ImageType =
    typename std::conditional_t<std::is_same_v<TImageSequence, CPUImageSequenceType>,
                                itk::Image<typename TImageSequence::PixelType, TImageSequence::ImageDimension - 1>,
                                itk::CudaImage<typename TImageSequence::PixelType, TImageSequence::ImageDimension - 1>>;
  using TVDenoisingFilterType = typename std::conditional_t<std::is_same_v<TImageSequence, CPUImageSequenceType>,
                                                            TotalVariationDenoisingBPDQImageFilter<ImageType>,
                                                            CudaTotalVariationDenoisingBPDQImageFilter>;
#else
  using ImageType = itk::Image<typename TImageSequence::PixelType, TImageSequence::ImageDimension - 1>;
  using TVDenoisingFilterType = rtk::TotalVariationDenoisingBPDQImageFilter<ImageType>;
#endif
  using ExtractFilterType = itk::ExtractImageFilter<TImageSequence, ImageType>;
  using PasteFilterType = itk::PasteImageFilter<TImageSequence, TImageSequence>;
  using CastFilterType = itk::CastImageFilter<ImageType, TImageSequence>;
  using ConstantImageSourceType = rtk::ConstantImageSource<TImageSequence>;

protected:
  TotalVariationDenoiseSequenceImageFilter();
  ~TotalVariationDenoiseSequenceImageFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  void
  GenerateOutputInformation() override;
  void
  GenerateInputRequestedRegion() override;

  /** Member pointers to the filters used internally (for convenience)*/
  typename TVDenoisingFilterType::Pointer   m_TVDenoisingFilter;
  typename ExtractFilterType::Pointer       m_ExtractFilter;
  typename PasteFilterType::Pointer         m_PasteFilter;
  typename CastFilterType::Pointer          m_CastFilter;
  typename ConstantImageSourceType::Pointer m_ConstantSource;

  /** Extraction regions for both extract filters */
  typename TImageSequence::RegionType m_ExtractAndPasteRegion;

  /** Information for the total variation denoising filter */
  double m_Gamma{ 1. };
  int    m_NumberOfIterations{ 1 };
  bool   m_DimensionsProcessed[ImageType::ImageDimension];
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkTotalVariationDenoiseSequenceImageFilter.hxx"
#endif

#endif
