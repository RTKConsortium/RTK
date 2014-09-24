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

#ifndef __rtkWarpSequenceImageFilter_h
#define __rtkWarpSequenceImageFilter_h

#include "rtkConstantImageSource.h"

#include <itkWarpImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkCastImageFilter.h>

namespace rtk
{
  /** \class WarpSequenceImageFilter
   * \brief Applies an N-D + time Motion Vector Field to an N-D + time sequence of images
   *
   * Most of the work in this filter is performed by the underlying itkWarpImageFilter.
   * The only difference is that this filter manages the last dimension specifically as time.
   *
   * \dot
   * digraph WarpSequenceImageFilter {
   *
   * Input0 [ label="Input 0 (Volume)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (Volume)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * ZeroMultiplyVolume [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * ZeroMultiplyGradient [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * BeforeZeroMultiplyVolume [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterGradient [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterZeroMultiplyGradient [label="", fixedsize="false", width=0, height=0, shape=none];
   * Gradient [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
   * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * AddGradient [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   * Divergence [ label="rtk::BackwardDifferenceDivergenceImageFilter" URL="\ref rtk::BackwardDifferenceDivergenceImageFilter"];
   * Multiply [ label="itk::MultiplyImageFilter (by beta)" URL="\ref itk::MultiplyImageFilter"];
   * SubtractVolume [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   * ConjugateGradient[ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
   * AfterConjugateGradient [label="", fixedsize="false", width=0, height=0, shape=none];
   * GradientTwo [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
   * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   * TVSoftThreshold [ label="rtk::SoftThresholdTVImageFilter" URL="\ref rtk::SoftThresholdTVImageFilter"];
   * BeforeTVSoftThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterTVSoftThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
   * SubtractTwo [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   *
   * Input0 -> BeforeZeroMultiplyVolume [arrowhead=None];
   * BeforeZeroMultiplyVolume -> ZeroMultiplyVolume;
   * BeforeZeroMultiplyVolume -> Gradient;
   * BeforeZeroMultiplyVolume -> ConjugateGradient;
   * Input1 -> Displaced;
   * Displaced -> BackProjection;
   * Gradient -> AfterGradient [arrowhead=None];
   * AfterGradient -> AddGradient;
   * AfterGradient -> ZeroMultiplyGradient;
   * ZeroMultiplyGradient -> AfterZeroMultiplyGradient [arrowhead=None];
   * AfterZeroMultiplyGradient -> AddGradient;
   * AfterZeroMultiplyGradient -> Subtract;
   * AddGradient -> Divergence;
   * Divergence -> Multiply;
   * Multiply -> SubtractVolume;
   * BackProjection -> SubtractVolume;
   * SubtractVolume -> ConjugateGradient;
   * ConjugateGradient -> AfterConjugateGradient;
   * AfterConjugateGradient -> GradientTwo;
   * GradientTwo -> Subtract;
   * Subtract -> BeforeTVSoftThreshold [arrowhead=None];
   * BeforeTVSoftThreshold -> TVSoftThreshold;
   * BeforeTVSoftThreshold -> SubtractTwo;
   * TVSoftThreshold -> AfterTVSoftThreshold [arrowhead=None];
   * AfterTVSoftThreshold -> SubtractTwo;
   *
   * AfterTVSoftThreshold -> AfterGradient [style=dashed];
   * SubtractTwo -> AfterZeroMultiplyGradient [style=dashed];
   * AfterConjugateGradient -> BeforeZeroMultiplyVolume [style=dashed];
   * AfterConjugateGradient -> Output [style=dashed];
   * }
   * \enddot
   *
   * \test rtk??.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TImageSequence,
          typename TMVFImageSequence = itk::Image< itk::CovariantVector < typename TImageSequence::ValueType,
                                                                          TImageSequence::ImageDimension-1 >,
                                                   TImageSequence::ImageDimension >,
          typename TImage = itk::Image< typename TImageSequence::ValueType,
                                        TImageSequence::ImageDimension-1 >,
          typename TMVFImage = itk::Image<itk::CovariantVector < typename TImageSequence::ValueType,
                                                                 TImageSequence::ImageDimension - 1 >,
                                          TImageSequence::ImageDimension - 1> >
class WarpSequenceImageFilter : public itk::ImageToImageFilter<TImageSequence, TImageSequence>
{
public:
    /** Standard class typedefs. */
    typedef WarpSequenceImageFilter                                  Self;
    typedef itk::ImageToImageFilter<TImageSequence, TImageSequence>  Superclass;
    typedef itk::SmartPointer< Self >                                Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(WarpSequenceImageFilter, IterativeConeBeamReconstructionFilter)

    /** Set the motion vector field used in input 1 */
    void SetDisplacementField(const TMVFImageSequence* MVFs);

    /** Get the motion vector field used in input 1 */
    typename TMVFImageSequence::Pointer GetDisplacementField();

    /** Typedefs of internal filters */
    typedef itk::WarpImageFilter<TImage, TImage, TMVFImage>         WarpFilterType;
    typedef itk::LinearInterpolateImageFunction<TImage, double >    InterpolatorType;
    typedef itk::ExtractImageFilter<TImageSequence, TImage>         ExtractFilterType;
    typedef itk::ExtractImageFilter<TMVFImageSequence, TMVFImage>   ExtractMVFFilterType;
    typedef itk::PasteImageFilter<TImageSequence,TImageSequence>    PasteFilterType;
    typedef itk::CastImageFilter<TImage, TImageSequence>            CastFilterType;
    typedef rtk::ConstantImageSource<TImageSequence>                ConstantImageSourceType;

protected:
    WarpSequenceImageFilter();
    ~WarpSequenceImageFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename WarpFilterType::Pointer          m_WarpFilter;
    typename ExtractFilterType::Pointer       m_ExtractFilter;
    typename ExtractMVFFilterType::Pointer    m_ExtractMVFFilter;
    typename PasteFilterType::Pointer         m_PasteFilter;
    typename CastFilterType::Pointer          m_CastFilter;
    typename ConstantImageSourceType::Pointer m_ConstantSource;

    /** Extraction regions for both extract filters */
    typename TImageSequence::RegionType       m_ExtractAndPasteRegion;
    typename TMVFImageSequence::RegionType    m_ExtractMVFRegion;

    /** The inputs of this filter have the same type (float, 3) but not the same meaning
    * It is normal that they do not occupy the same physical space. Therefore this check
    * must be removed */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateOutputInformation();

private:
    WarpSequenceImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWarpSequenceImageFilter.txx"
#endif

#endif
