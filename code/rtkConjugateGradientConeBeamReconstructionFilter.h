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

#ifndef __rtkConjugateGradientConeBeamReconstructionFilter_h
#define __rtkConjugateGradientConeBeamReconstructionFilter_h

#include <itkMultiplyImageFilter.h>
#include <itkTimeProbe.h>
#include <itkDivideOrZeroOutImageFilter.h>

#include "rtkConjugateGradientImageFilter.h"
#include "rtkReconstructionConjugateGradientOperator.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkDisplacedDetectorImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkLaplacianImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "rtkCudaConjugateGradientImageFilter_3f.h"
  #include "rtkCudaDisplacedDetectorImageFilter.h"
  #include "rtkCudaConstantVolumeSource.h"
#endif

namespace rtk
{
  /** \class ConjugateGradientConeBeamReconstructionFilter
   * \brief Implements ConjugateGradient
   *
   * This filter implements the ConjugateGradient method.
   * ConjugateGradient attempts to find the f that minimizes
   * || sqrt(D) (Rf -p) ||_2^2 + gamma || grad f ||_2^2
   * with R the forward projection operator,
   * p the measured projections, and D the displaced detector weighting operator.
   *
   * With gamma=0, this it is similar to the ART and SART methods. The difference lies
   * in the algorithm employed to minimize this cost function. ART uses the
   * Kaczmarz method (projects and back projects one ray at a time),
   * SART the block-Kaczmarz method (projects and back projects one projection
   * at a time), and ConjugateGradient a conjugate gradient method
   * (projects and back projects all projections together).
   *
   * With gamma > 0, a regularization is applied.
   *
   * \dot
   * digraph ConjugateGradientConeBeamReconstructionFilter {
   *
   * Input0 [ label="Input 0 (Volume)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Input2 [label="Input 2 (Weights)"];
   * Input2 [shape=Mdiamond];
   * Output [label="Output (Volume)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * MultiplyProjections [label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
   * MultiplyVolumes [label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
   * MultiplyOutput [label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
   * ConjugateGradient[ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
   * VolumeSource [ label="rtk::ConstantImageSource (Volume)" URL="\ref rtk::ConstantImageSource"];
   * ProjectionsSource [ label="rtk::ConstantImageSource (Projections)" URL="\ref rtk::ConstantImageSource"];
   * BackProjForPreconditioning [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * BackProjForNormalization [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * Divide [label="itk::DivideOrZeroOutImageFilter" URL="\ref itk::DivideOrZeroOutImageFilter"];
   *
   * AfterVolumeSource [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterDivide [label="Preconditioning weights", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input0 -> ConjugateGradient;
   * Input1 -> Displaced;
   * Input2 -> MultiplyProjections;
   * Input2 -> BackProjForPreconditioning;
   * Displaced -> MultiplyProjections;
   * MultiplyProjections -> BackProjection;
   * VolumeSource -> AfterVolumeSource [arrowhead=none];
   * AfterVolumeSource -> BackProjection;
   * AfterVolumeSource -> BackProjForPreconditioning;
   * AfterVolumeSource -> BackProjForNormalization;
   * ProjectionsSource -> BackProjForNormalization;
   * BackProjForPreconditioning -> Divide;
   * BackProjForNormalization -> Divide;
   * Divide -> AfterDivide [arrowhead=none];
   * AfterDivide -> MultiplyVolumes;
   * AfterDivide -> MultiplyOutput;
   * BackProjection -> MultiplyVolumes;
   * MultiplyVolumes -> ConjugateGradient;
   * ConjugateGradient -> MultiplyOutput;
   * MultiplyOutput -> Output;
   * }
   * \enddot
   *
   * \test rtkconjugategradientreconstructiontest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TOutputImage >
class ConjugateGradientConeBeamReconstructionFilter : public rtk::IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>
{
public:
    /** Standard class typedefs. */
    typedef ConjugateGradientConeBeamReconstructionFilter                      Self;
    typedef IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>  Superclass;
    typedef itk::SmartPointer< Self >                                          Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConjugateGradientConeBeamReconstructionFilter, itk::ImageToImageFilter)

    /** The 3D image to be updated */
    void SetInputVolume(const TOutputImage* Volume);

    /** The gated measured projections */
    void SetInputProjectionStack(const TOutputImage* Projection);

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage >  ForwardProjectionFilterType;
    typedef typename ForwardProjectionFilterType::Pointer                    ForwardProjectionFilterPointer;
    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >     BackProjectionFilterType;
    typedef rtk::ConjugateGradientImageFilter<TOutputImage>                  ConjugateGradientFilterType;
    typedef itk::MultiplyImageFilter<TOutputImage>                           MultiplyFilterType;
    typedef rtk::ReconstructionConjugateGradientOperator<TOutputImage>       CGOperatorFilterType;
    typedef rtk::DisplacedDetectorImageFilter<TOutputImage>                  DisplacedDetectorFilterType;
    typedef rtk::ConstantImageSource<TOutputImage>                           ConstantImageSourceType;
    typedef itk::DivideOrZeroOutImageFilter<TOutputImage>                    DivideFilterType;

    /** Pass the ForwardProjection filter to the conjugate gradient operator */
    void SetForwardProjectionFilter (int _arg);

    /** Pass the backprojection filter to the conjugate gradient operator and to the back projection filter generating the B of AX=B */
    void SetBackProjectionFilter (int _arg);

    /** Pass the geometry to all filters needing it */
    itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

    itkSetMacro(NumberOfIterations, int)
    itkGetMacro(NumberOfIterations, int)

    itkSetMacro(MeasureExecutionTimes, bool)
    itkGetMacro(MeasureExecutionTimes, bool)

    /** If Weighted, perform weighted least squares optimization instead of unweighted */
    itkSetMacro(Weighted, bool)
    itkGetMacro(Weighted, bool)

    /** If Weighted and Preconditioned, computes preconditioning weights to speed up CG convergence */
    itkSetMacro(Preconditioned, bool)
    itkGetMacro(Preconditioned, bool)
    
    /** If Regularized, perform laplacian-based regularization during 
     *  reconstruction (gamma is the strength of the regularization) */
    itkSetMacro(Regularized, bool)
    itkGetMacro(Regularized, bool)
    itkSetMacro(Gamma, float)
    itkGetMacro(Gamma, float)

protected:
    ConjugateGradientConeBeamReconstructionFilter();
    ~ConjugateGradientConeBeamReconstructionFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename MultiplyFilterType::Pointer                                        m_MultiplyProjectionsFilter;
    typename MultiplyFilterType::Pointer                                        m_MultiplyVolumeFilter;
    typename MultiplyFilterType::Pointer                                        m_MultiplyOutputFilter;
    typename ConjugateGradientFilterType::Pointer                               m_ConjugateGradientFilter;
    typename CGOperatorFilterType::Pointer                                      m_CGOperator;
    typename ForwardProjectionImageFilter<TOutputImage, TOutputImage>::Pointer  m_ForwardProjectionFilter;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilter;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilterForB;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilterForPreconditioning;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilterForNormalization;
    typename DisplacedDetectorFilterType::Pointer                               m_DisplacedDetectorFilter;
    typename ConstantImageSourceType::Pointer                                   m_ConstantVolumeSource;
    typename ConstantImageSourceType::Pointer                                   m_ConstantProjectionsSource;
    typename DivideFilterType::Pointer                                          m_DivideFilter;

    /** The inputs of this filter have the same type (float, 3) but not the same meaning
    * It is normal that they do not occupy the same physical space. Therefore this check
    * must be removed */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion();
    void GenerateOutputInformation();

private:
    ConjugateGradientConeBeamReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    ThreeDCircularProjectionGeometry::Pointer m_Geometry;

    int   m_NumberOfIterations;
    float m_Gamma;
    bool  m_MeasureExecutionTimes;
    bool  m_Weighted;
    bool  m_Preconditioned;
    bool  m_Regularized;
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientConeBeamReconstructionFilter.txx"
#endif

#endif
