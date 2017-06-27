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

#ifndef rtkUnwarpSequenceConjugateGradientOperator_h
#define rtkUnwarpSequenceConjugateGradientOperator_h

#include "rtkWarpSequenceImageFilter.h"
#include "rtkConjugateGradientOperator.h"

namespace rtk
{

  /** \class UnwarpSequenceConjugateGradientOperator
   * \brief Implements the operator A used in the conjugate gradient unwarp sequence filter
   *
   * \dot
   * digraph UnwarpSequenceConjugateGradientOperator {
   *
   * Input0 [ label="Input 0 (4D volume sequence)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (4D DVF)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (4D volume sequence)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * WarpSequenceBackward [ label="rtk::WarpSequenceImageFilter" URL="\ref rtk::WarpSequenceImageFilter"];
   * WarpSequenceForward [ label="rtk::WarpSequenceImageFilter (forward)" URL="\ref rtk::WarpSequenceImageFilter"];
   *
   * Input0 -> WarpSequenceBackward;
   * WarpSequenceBackward -> WarpSequenceForward;
   * WarpSequenceForward -> Output;
   *
   * }
   * \enddot
   *
   * \test rtkunwarpsequencetest.cxx
   *
   * \author Cyril Mory
   */

  template< typename TImageSequence,
            typename TDVFImageSequence = itk::Image< itk::CovariantVector < typename TImageSequence::ValueType,
                                                                            TImageSequence::ImageDimension-1 >,
                                                     TImageSequence::ImageDimension >,
            typename TImage = itk::Image< typename TImageSequence::ValueType,
                                          TImageSequence::ImageDimension-1 >,
            typename TDVFImage = itk::Image<itk::CovariantVector < typename TImageSequence::ValueType,
                                                                   TImageSequence::ImageDimension - 1 >,
                                            TImageSequence::ImageDimension - 1> >
class UnwarpSequenceConjugateGradientOperator : public ConjugateGradientOperator< TImageSequence >
{
public:
    /** Standard class typedefs. */
    typedef UnwarpSequenceConjugateGradientOperator       Self;
    typedef ConjugateGradientOperator< TImageSequence >   Superclass;
    typedef itk::SmartPointer< Self >                     Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(rtkUnwarpSequenceConjugateGradientOperator, ConjugateGradientOperator)

    typedef rtk::WarpSequenceImageFilter<TImageSequence, TDVFImageSequence, TImage, TDVFImage> WarpSequenceFilterType;

    /** Set the motion vector field used in input 1 */
    void SetDisplacementField(const TDVFImageSequence* DVFs);

    /** Get the motion vector field used in input 1 */
    typename TDVFImageSequence::Pointer GetDisplacementField();

    /** Phase shift to simulate phase estimation errors */
    itkSetMacro(PhaseShift, float)
    itkGetMacro(PhaseShift, float)

    itkSetMacro(UseNearestNeighborInterpolationInWarping, bool)
    itkGetMacro(UseNearestNeighborInterpolationInWarping, bool)

    /** Set and Get for the UseCudaCyclicDeformation variable */
    itkSetMacro(UseCudaCyclicDeformation, bool)
    itkGetMacro(UseCudaCyclicDeformation, bool)

protected:
    UnwarpSequenceConjugateGradientOperator();
    ~UnwarpSequenceConjugateGradientOperator() {}

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    /** Member pointers to the filters used internally (for convenience)*/
    typename WarpSequenceFilterType::Pointer              m_WarpSequenceBackwardFilter;
    typename WarpSequenceFilterType::Pointer              m_WarpSequenceForwardFilter;

    float m_PhaseShift;
    bool  m_UseNearestNeighborInterpolationInWarping; //Default is false, linear interpolation is used instead

    /** When the inputs have the same type, ITK checks whether they occupy the
    * same physical space or not. Obviously they dont, so we have to remove this check
    */
    void VerifyInputInformation() ITK_OVERRIDE {}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion() ITK_OVERRIDE;
    void GenerateOutputInformation() ITK_OVERRIDE;

    bool m_UseCudaCyclicDeformation;

private:
    UnwarpSequenceConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkUnwarpSequenceConjugateGradientOperator.hxx"
#endif

#endif
