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

#ifndef __rtkUnwarpSequenceImageFilter_h
#define __rtkUnwarpSequenceImageFilter_h

#include <itkMultiplyImageFilter.h>

#include "rtkConjugateGradientImageFilter.h"
#include "rtkUnwarpSequenceConjugateGradientOperator.h"
#include "rtkWarpSequenceImageFilter.h"

namespace rtk
{
  /** \class UnwarpSequenceImageFilter
   * \brief Finds the image sequence that, once warped, equals the input image sequence.
   *
   * This filter attempts to invert a deformation by Conjugate Gradient optimization.
   *
   * \dot
   * digraph UnwarpSequenceImageFilter {
   *
   * Input0 [ label="Input 0 (4D volume sequence)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (4D DVF)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (4D volume sequence)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * ZeroMultiplyVolumeSequence [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * WarpSequenceForward [label="rtk::WarpSequenceImageFilter (forward)" URL="\ref rtk::WarpSequenceImageFilter"];
   * ConjugateGradient[ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
   * AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterInput1 [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input0 -> AfterInput0 [arrowhead=none];
   * AfterInput0 -> ZeroMultiplyVolumeSequence;
   * AfterInput0 -> WarpSequenceForward;
   * Input1 -> AfterInput1;
   * AfterInput1 -> WarpSequenceForward;
   * AfterInput1 -> ConjugateGradient;
   * ZeroMultiplyVolumeSequence -> ConjugateGradient;
   * WarpSequenceForward -> ConjugateGradient;
   * ConjugateGradient -> Output;
   *
   * }
   * \enddot
   *
   * \test rtkunwarpsequencetest.cxx
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
class UnwarpSequenceImageFilter : public itk::ImageToImageFilter<TImageSequence, TImageSequence>
{
public:
    /** Standard class typedefs. */
    typedef UnwarpSequenceImageFilter                                 Self;
    typedef itk::ImageToImageFilter<TImageSequence, TImageSequence>   Superclass;
    typedef itk::SmartPointer< Self >                                 Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(UnwarpSequenceImageFilter, ImageToImageFilter)

    typedef itk::MultiplyImageFilter<TImageSequence>                          MultiplySequenceFilterType;
    typedef rtk::UnwarpSequenceConjugateGradientOperator<TImageSequence,
                                                         TMVFImageSequence,
                                                         TImage,
                                                         TMVFImage>           CGOperatorFilterType;
    typedef rtk::WarpSequenceImageFilter< TImageSequence,
                                          TMVFImageSequence,
                                          TImage,
                                          TMVFImage>                          WarpForwardFilterType;
    typedef rtk::ConjugateGradientImageFilter<TImageSequence>                 ConjugateGradientFilterType;

    /** Set the motion vector field used in input 1 */
    void SetDisplacementField(const TMVFImageSequence* MVFs);

    /** Get the motion vector field used in input 1 */
    typename TMVFImageSequence::Pointer GetDisplacementField();

    /** Number of conjugate gradient iterations */
    itkSetMacro(NumberOfIterations, float)
    itkGetMacro(NumberOfIterations, float)

    /** Phase shift to simulate phase estimation errors */
    itkSetMacro(PhaseShift, float)
    itkGetMacro(PhaseShift, float)

protected:
    UnwarpSequenceImageFilter();
    ~UnwarpSequenceImageFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename MultiplySequenceFilterType::Pointer                    m_ZeroMultiplySequenceFilter;
    typename ConjugateGradientFilterType::Pointer                   m_ConjugateGradientFilter;
    typename CGOperatorFilterType::Pointer                          m_CGOperator;
    typename WarpForwardFilterType::Pointer                         m_WarpForwardFilter;

    /** Member variables */
    float m_PhaseShift;

    /** The inputs of this filter have the same type (float, 3) but not the same meaning
    * It is normal that they do not occupy the same physical space. Therefore this check
    * must be removed */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion();
    void GenerateOutputInformation();

private:
    UnwarpSequenceImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    unsigned int    m_NumberOfIterations;
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkUnwarpSequenceImageFilter.txx"
#endif

#endif
