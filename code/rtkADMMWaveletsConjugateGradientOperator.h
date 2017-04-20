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

#ifndef rtkADMMWaveletsConjugateGradientOperator_h
#define rtkADMMWaveletsConjugateGradientOperator_h

#include <itkMultiplyImageFilter.h>
#include <itkAddImageFilter.h>

#include "rtkConjugateGradientOperator.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkDisplacedDetectorImageFilter.h"

#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

  /** \class ADMMWaveletsConjugateGradientOperator
   * \brief Implements the operator A used in the conjugate gradient step
   * of ADMM reconstruction with wavelets regularization
   *
   * This filter implements the operator A used in the conjugate gradient step
   * of a reconstruction method based on compressed sensing. The method attempts
   * to find the f that minimizes || Rf -p ||_2^2 + alpha * || W(f) ||_1, with R the
   * forward projection operator, p the measured projections, and W the
   * Daubechies wavelets transform.
   * Details on the method and the calculations can be found in
   *
   * Mory, C., B. Zhang, V. Auvray, M. Grass, D. Schafer, F. Peyrin, S. Rit, P. Douek,
   * and L. Boussel. “ECG-Gated C-Arm Computed Tomography Using L1 Regularization.”
   * In Proceedings of the 20th European Signal Processing Conference (EUSIPCO), 2728–32, 2012.
   *
   * This filter takes in input f and outputs R_t R f + beta * W_t W f. The Daubechies
   * wavelets being orthogonal, W_t happens to be the inverse of W, and therefore
   * the filter outputs R_t R f + beta * f
   *
   * \dot
   * digraph ADMMWaveletsConjugateGradientOperator {
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
   * Add [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   * Multiply [ label="itk::MultiplyImageFilter (by lambda)" URL="\ref itk::MultiplyImageFilter"];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
   *
   * Input0 -> BeforeZeroMultiplyVolume [arrowhead=none];
   * BeforeZeroMultiplyVolume -> ZeroMultiplyVolume;
   * BeforeZeroMultiplyVolume -> ForwardProjection;
   * BeforeZeroMultiplyVolume -> Multiply;
   * Input1 -> ZeroMultiplyProjections;
   * ZeroMultiplyProjections -> ForwardProjection;
   * ZeroMultiplyVolume -> BackProjection;
   * ForwardProjection -> Displaced;
   * Displaced -> BackProjection;
   * BackProjection -> Add;
   * Multiply -> Add;
   * Add -> Output;
   *
   * }
   * \enddot
   *
   * \test rtkadmmWaveletstest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TOutputImage >
class ADMMWaveletsConjugateGradientOperator : public ConjugateGradientOperator< TOutputImage >
{
public:
    /** Standard class typedefs. */
    typedef ADMMWaveletsConjugateGradientOperator         Self;
    typedef ConjugateGradientOperator< TOutputImage >     Superclass;
    typedef itk::SmartPointer< Self >                     Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(rtkADMMWaveletsConjugateGradientOperator, ConjugateGradientOperator)

    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >    BackProjectionFilterType;
    typedef typename BackProjectionFilterType::Pointer                      BackProjectionFilterPointer;

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage > ForwardProjectionFilterType;
    typedef typename ForwardProjectionFilterType::Pointer                   ForwardProjectionFilterPointer;

    typedef itk::MultiplyImageFilter<TOutputImage>                          MultiplyFilterType;
    typedef itk::AddImageFilter<TOutputImage>                               AddFilterType;

    typedef rtk::DisplacedDetectorImageFilter<TOutputImage>                 DisplacedDetectorFilterType;

    /** Set the backprojection filter*/
    void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

    /** Set the forward projection filter*/
    void SetForwardProjectionFilter (const ForwardProjectionFilterPointer _arg);

    /** Set the geometry of both m_BackProjectionFilter and m_ForwardProjectionFilter */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

    /** Set the regularization parameter */
    itkSetMacro(Beta, float)

    /** Set / Get whether the displaced detector filter should be disabled */
    itkSetMacro(DisableDisplacedDetectorFilter, bool)
    itkGetMacro(DisableDisplacedDetectorFilter, bool)

protected:
    ADMMWaveletsConjugateGradientOperator();
    ~ADMMWaveletsConjugateGradientOperator() {}

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    /** Member pointers to the filters used internally (for convenience)*/
    BackProjectionFilterPointer            m_BackProjectionFilter;
    ForwardProjectionFilterPointer         m_ForwardProjectionFilter;

    typename AddFilterType::Pointer                   m_AddFilter;
    typename MultiplyFilterType::Pointer              m_MultiplyFilter;
    typename MultiplyFilterType::Pointer              m_ZeroMultiplyProjectionFilter;
    typename MultiplyFilterType::Pointer              m_ZeroMultiplyVolumeFilter;
    typename DisplacedDetectorFilterType::Pointer     m_DisplacedDetectorFilter;

    float m_Beta;
    bool  m_DisableDisplacedDetectorFilter;

    /** When the inputs have the same type, ITK checks whether they occupy the
    * same physical space or not. Obviously they dont, so we have to remove this check
    */
    void VerifyInputInformation() ITK_OVERRIDE {}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion() ITK_OVERRIDE;
    void GenerateOutputInformation() ITK_OVERRIDE;

private:
    ADMMWaveletsConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkADMMWaveletsConjugateGradientOperator.hxx"
#endif

#endif
