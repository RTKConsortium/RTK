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

#ifndef __rtkReconstructionConjugateGradientOperator_h
#define __rtkReconstructionConjugateGradientOperator_h

#include <itkMultiplyImageFilter.h>

#include "rtkConjugateGradientOperator.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkDisplacedDetectorImageFilter.h"

namespace rtk
{

  /** \class ReconstructionConjugateGradientOperator
   * \brief Implements the operator A used in conjugate gradient reconstruction
   *
   * This filter implements the operator A used in the conjugate gradient reconstruction method,
   * which attempts to find the f that minimizes
   * || sqrt(D) (Rf -p) ||_2^2,
   * with R the forward projection operator,
   * p the measured projections, and D the displaced detector weighting operator.
   * In this it is similar to the ART and SART methods. The difference lies
   * in the algorithm employed to minimize this cost function. ART uses the
   * Kaczmarz method (projects and back projects one ray at a time),
   * SART the block-Kaczmarz method (projects and back projects one projection
   * at a time), and ConjugateGradient a conjugate gradient method
   * (projects and back projects all projections together).
   *
   * This filter takes in input f and outputs R_t D R f
   *
   * \dot
   * digraph ReconstructionConjugateGradientOperator {
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
   * ZeroMultiplyProjections [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * BeforeZeroMultiplyVolume [label="", fixedsize="false", width=0, height=0, shape=none];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
   *
   * Input0 -> BeforeZeroMultiplyVolume [arrowhead=None];
   * BeforeZeroMultiplyVolume -> ZeroMultiplyVolume;
   * BeforeZeroMultiplyVolume -> ForwardProjection;
   * Input1 -> ZeroMultiplyProjections;
   * ZeroMultiplyProjections -> ForwardProjection;
   * ZeroMultiplyVolume -> BackProjection;
   * ForwardProjection -> Displaced;
   * Displaced -> BackProjection;
   * BackProjection -> Output;
   *
   * }
   * \enddot
   *
   * \test rtkconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TOutputImage >
class ReconstructionConjugateGradientOperator : public ConjugateGradientOperator< TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef ReconstructionConjugateGradientOperator    Self;
  typedef ConjugateGradientOperator< TOutputImage >  Superclass;
  typedef itk::SmartPointer< Self >                  Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(rtkReconstructionConjugateGradientOperator, ConjugateGradientOperator)

  typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >    BackProjectionFilterType;
  typedef typename BackProjectionFilterType::Pointer                      BackProjectionFilterPointer;

  typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage > ForwardProjectionFilterType;
  typedef typename ForwardProjectionFilterType::Pointer                   ForwardProjectionFilterPointer;

  typedef rtk::DisplacedDetectorImageFilter<TOutputImage>                 DisplacedDetectorFilterType;
  typedef itk::MultiplyImageFilter<TOutputImage>                          MultiplyFilterType;

  /** Set the backprojection filter*/
  void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

  /** Set the forward projection filter*/
  void SetForwardProjectionFilter (const ForwardProjectionFilterPointer _arg);

  /** Set the geometry of both m_BackProjectionFilter and m_ForwardProjectionFilter */
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

protected:
  ReconstructionConjugateGradientOperator();
  ~ReconstructionConjugateGradientOperator(){}

  /** Does the real work. */
  virtual void GenerateData();

  /** Member pointers to the filters used internally (for convenience)*/
  BackProjectionFilterPointer            m_BackProjectionFilter;
  ForwardProjectionFilterPointer         m_ForwardProjectionFilter;

  typename MultiplyFilterType::Pointer              m_MultiplyFilter;
  typename MultiplyFilterType::Pointer              m_ZeroMultiplyProjectionFilter;
  typename MultiplyFilterType::Pointer              m_ZeroMultiplyVolumeFilter;
  typename DisplacedDetectorFilterType::Pointer     m_DisplacedDetectorFilter;

  /** Member attributes */
  rtk::ThreeDCircularProjectionGeometry::Pointer    m_Geometry;

  /** When the inputs have the same type, ITK checks whether they occupy the
   * same physical space or not. Obviously they dont, so we have to remove this check */
  void VerifyInputInformation(){}

  /** The volume and the projections must have different requested regions */
  void GenerateInputRequestedRegion();
  void GenerateOutputInformation();

private:
  ReconstructionConjugateGradientOperator(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkReconstructionConjugateGradientOperator.txx"
#endif

#endif
