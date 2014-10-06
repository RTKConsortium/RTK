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
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkForwardWarpImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkCastImageFilter.h>

#ifdef RTK_USE_CUDA
  #include "rtkCudaWarpImageFilter.h"
#else
  #include <itkWarpImageFilter.h>
#endif

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
   * Input0 [ label="Input 0 (Sequence of images)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Sequence of MVFs)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (Sequence of images)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * Extract [label="itk::ExtractImageFilter (for images)" URL="\ref itk::ExtractImageFilter"];
   * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for MVFs)" URL="\ref rtk::CyclicDeformationImageFilter"];
   * Warp [ label="itk::WarpImageFilter" URL="\ref itk::WarpImageFilter"];
   * Cast [ label="itk::CastImageFilter" URL="\ref itk::CastImageFilter"];
   * Paste [ label="itk::PasteImageFilter" URL="\ref rtk::PasteImageFilter"];
   * ConstantSource [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
   * BeforePaste [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterPaste [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input0 -> Extract;
   * Input1 -> CyclicDeformation;
   * Extract -> Warp;
   * CyclicDeformation -> Warp;
   * Warp -> Cast;
   * Cast -> BeforePaste [arrowhead=None];
   * BeforePaste -> Paste;
   * Paste -> AfterPaste [arrowhead=None];
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

    /** Set/Get for m_ForwardWarp */
    itkGetMacro(ForwardWarp, bool)
    itkSetMacro(ForwardWarp, bool)

    /** Typedefs of internal filters */
#ifdef RTK_USE_CUDA
    typedef rtk::CudaWarpImageFilter                                CudaWarpFilterType;
#endif
    typedef itk::WarpImageFilter<TImage, TImage, TMVFImage>         WarpFilterType;
    typedef rtk::ForwardWarpImageFilter<TImage, TImage, TMVFImage>  ForwardWarpFilterType;

    typedef itk::LinearInterpolateImageFunction<TImage, double >    InterpolatorType;
    typedef itk::ExtractImageFilter<TImageSequence, TImage>         ExtractFilterType;
    typedef rtk::CyclicDeformationImageFilter<TMVFImage>            MVFInterpolatorType;
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
    typename MVFInterpolatorType::Pointer     m_MVFInterpolatorFilter;
    typename PasteFilterType::Pointer         m_PasteFilter;
    typename CastFilterType::Pointer          m_CastFilter;
    typename ConstantImageSourceType::Pointer m_ConstantSource;

    /** Extraction regions for both extract filters */
    typename TImageSequence::RegionType       m_ExtractAndPasteRegion;

    /** Perform a forward warping (using splat) instead of the standard backward warping */
    bool m_ForwardWarp;

    /** The inputs of this filter have the same type (float, 3) but not the same meaning
    * It is normal that they do not occupy the same physical space. Therefore this check
    * must be removed */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateOutputInformation();
    void GenerateInputRequestedRegion();

private:
    WarpSequenceImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWarpSequenceImageFilter.txx"
#endif

#endif
