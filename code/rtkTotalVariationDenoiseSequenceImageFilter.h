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

#ifndef rtkTotalVariationDenoiseSequenceImageFilter_h
#define rtkTotalVariationDenoiseSequenceImageFilter_h

#include "rtkConstantImageSource.h"

#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkCastImageFilter.h>

#ifdef RTK_USE_CUDA
  #include "rtkCudaTotalVariationDenoisingBPDQImageFilter.h"
#else
  #include "rtkTotalVariationDenoisingBPDQImageFilter.h"
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
   * TVDenoising [ label="rtk::TotalVariationDenoisingBPDQImageFilter" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
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
   * \ingroup ReconstructionAlgorithm
   */

template< typename TImageSequence >
class TotalVariationDenoiseSequenceImageFilter : public itk::ImageToImageFilter<TImageSequence, TImageSequence>
{
public:
    /** Standard class typedefs. */
    typedef TotalVariationDenoiseSequenceImageFilter                 Self;
    typedef itk::ImageToImageFilter<TImageSequence, TImageSequence>  Superclass;
    typedef itk::SmartPointer< Self >                                Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(TotalVariationDenoiseSequenceImageFilter, ImageToImageFilter)

    /** Set/Get for the TotalVariationDenoisingBPDQImageFilter */
    itkGetMacro(Gamma, double)
    itkSetMacro(Gamma, double)

    itkGetMacro(NumberOfIterations, int)
    itkSetMacro(NumberOfIterations, int)

    void SetDimensionsProcessed(bool* arg);

    /** Typedefs of internal filters */
#ifdef RTK_USE_CUDA
    typedef itk::CudaImage<typename TImageSequence::PixelType, TImageSequence::ImageDimension - 1>   TImage;
    typedef rtk::CudaTotalVariationDenoisingBPDQImageFilter                             TVDenoisingFilterType;
#else
    typedef itk::Image<typename TImageSequence::PixelType, TImageSequence::ImageDimension - 1>   TImage;
    typedef rtk::TotalVariationDenoisingBPDQImageFilter<TImage>                         TVDenoisingFilterType;
#endif
    typedef itk::ExtractImageFilter<TImageSequence, TImage>         ExtractFilterType;
    typedef itk::PasteImageFilter<TImageSequence,TImageSequence>    PasteFilterType;
    typedef itk::CastImageFilter<TImage, TImageSequence>            CastFilterType;
    typedef rtk::ConstantImageSource<TImageSequence>                ConstantImageSourceType;

protected:
    TotalVariationDenoiseSequenceImageFilter();
    ~TotalVariationDenoiseSequenceImageFilter() {}

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    void GenerateOutputInformation() ITK_OVERRIDE;
    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Member pointers to the filters used internally (for convenience)*/
    typename TVDenoisingFilterType::Pointer   m_TVDenoisingFilter;
    typename ExtractFilterType::Pointer       m_ExtractFilter;
    typename PasteFilterType::Pointer         m_PasteFilter;
    typename CastFilterType::Pointer          m_CastFilter;
    typename ConstantImageSourceType::Pointer m_ConstantSource;

    /** Extraction regions for both extract filters */
    typename TImageSequence::RegionType       m_ExtractAndPasteRegion;

    /** Information for the total variation denoising filter */
    double m_Gamma;
    int    m_NumberOfIterations;
    bool   m_DimensionsProcessed[TImage::ImageDimension];

private:
    TotalVariationDenoiseSequenceImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkTotalVariationDenoiseSequenceImageFilter.hxx"
#endif

#endif
