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
#ifndef rtkFourDReconstructionConjugateGradientOperator_h
#define rtkFourDReconstructionConjugateGradientOperator_h

#include "rtkConjugateGradientOperator.h"

#include <itkArray2D.h>
#include <itkMultiplyImageFilter.h>

#include "rtkConstantImageSource.h"
#include "rtkInterpolatorWithKnownWeightsImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkSplatWithKnownWeightsImageFilter.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkDisplacedDetectorImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaInterpolateImageFilter.h"
#  include "rtkCudaSplatImageFilter.h"
#  include "rtkCudaConstantVolumeSource.h"
#  include "rtkCudaConstantVolumeSeriesSource.h"
#  include "rtkCudaDisplacedDetectorImageFilter.h"
#endif

namespace rtk
{
  /** \class FourDReconstructionConjugateGradientOperator
   * \brief Implements part of the 4D reconstruction by conjugate gradient
   *
   * See the reference paper: "Cardiac C-arm computed tomography using
   * a 3D + time ROI reconstruction method with spatial and temporal regularization"
   * by Mory et al.
   *
   * 4D conjugate gradient reconstruction consists in iteratively
   * minimizing the following cost function:
   *
   * Sum_over_theta || sqrt(D) (R_theta S_theta f - p_theta) ||_2^2
   *
   * with
   * - f a 4D series of 3D volumes, each one being the reconstruction
   * at a given respiratory/cardiac phase
   * - p_theta is the projection measured at angle theta
   * - S_theta an interpolation operator which, from the 3D + time sequence f,
   * estimates the 3D volume through which projection p_theta has been acquired
   * - R_theta is the X-ray transform (the forward projection operator) for angle theta
   * - D the displaced detector weighting matrix
   *
   * Computing the gradient of this cost function yields:
   *
   * S_theta^T R_theta^T D R_theta S_theta f - S_theta^T R_theta^T D p_theta
   *
   * where A^T means the adjoint of operator A.
   *
   * FourDReconstructionConjugateGradientOperator implements S_theta^T R_theta^T D R_theta S_theta.
   * It can be achieved by a FourDToProjectionStackImageFilter followed by
   * a DisplacedDetectorFilter and ProjectionStackToFourDImageFilter (simple implementation), or
   * by assembling the internal pipelines of these filters, and removing
   * the unnecessary filters in the middle (a PasteImageFilter and an ExtractImageFilter), which
   * results in performance gain and easier GPU memory management.
   * The current implementation is the optimized one.
   *
   * \dot
   * digraph FourDReconstructionConjugateGradientOperator {
   *
   * Input0 [ label="Input 0 (Input: 4D sequence of volumes)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Input2 [label="Input 2 (Projection weights)"];
   * Input2 [shape=Mdiamond];
   * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * SourceVol [ label="rtk::ConstantImageSource (volume)" URL="\ref rtk::ConstantImageSource"];
   * SourceVol2 [ label="rtk::ConstantImageSource (volume)" URL="\ref rtk::ConstantImageSource"];
   * SourceProj [ label="rtk::ConstantImageSource (projections)" URL="\ref rtk::ConstantImageSource"];
   * Source4D [ label="rtk::ConstantImageSource (4D)" URL="\ref rtk::ConstantImageSource"];
   * ForwardProj [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * Interpolation [ label="InterpolatorWithKnownWeightsImageFilter" URL="\ref rtk::InterpolatorWithKnownWeightsImageFilter"];
   * BackProj [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * Splat [ label="rtk::SplatWithKnownWeightsImageFilter" URL="\ref rtk::SplatWithKnownWeightsImageFilter"];
   * AfterSplat [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterSource4D [label="", fixedsize="false", width=0, height=0, shape=none];
   * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
   *
   * Input0 -> Interpolation;
   * SourceVol -> Interpolation;
   * Interpolation -> ForwardProj;
   * SourceVol2 -> BackProj;
   * ForwardProj -> Displaced;
   * Displaced -> BackProj;
   * BackProj -> Splat;
   * Splat -> AfterSplat[arrowhead=none];
   * AfterSplat -> Output;
   * AfterSplat -> AfterSource4D[style=dashed, constraint=false];
   * Source4D -> AfterSource4D[arrowhead=none];
   * AfterSource4D -> Splat;
   * SourceProj -> ForwardProj;
   * }
   * \enddot
   *
   * \test rtkfourdconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename VolumeSeriesType, typename ProjectionStackType>
class FourDReconstructionConjugateGradientOperator : public ConjugateGradientOperator< VolumeSeriesType>
{
public:
    /** Standard class typedefs. */
    typedef FourDReconstructionConjugateGradientOperator        Self;
    typedef ConjugateGradientOperator< VolumeSeriesType>        Superclass;
    typedef itk::SmartPointer< Self >                           Pointer;

  /** Convenient typedef */
    typedef ProjectionStackType                                 VolumeType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(FourDReconstructionConjugateGradientOperator, ConjugateGradientOperator)

    /** Set/Get the 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);
    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();

    /** Set/Get the stack of projections */
    void SetInputProjectionStack(const ProjectionStackType* Projections);
    typename ProjectionStackType::ConstPointer GetInputProjectionStack();

    typedef rtk::BackProjectionImageFilter< ProjectionStackType, ProjectionStackType >          BackProjectionFilterType;
    typedef rtk::ForwardProjectionImageFilter< ProjectionStackType, ProjectionStackType >       ForwardProjectionFilterType;
    typedef rtk::InterpolatorWithKnownWeightsImageFilter<VolumeType, VolumeSeriesType>          InterpolationFilterType;
    typedef rtk::SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>                 SplatFilterType;
    typedef rtk::ConstantImageSource<VolumeType>                                                ConstantVolumeSourceType;
    typedef rtk::ConstantImageSource<ProjectionStackType>                                       ConstantProjectionStackSourceType;
    typedef rtk::ConstantImageSource<VolumeSeriesType>                                          ConstantVolumeSeriesSourceType;
    typedef rtk::DisplacedDetectorImageFilter<ProjectionStackType>                              DisplacedDetectorFilterType;

    /** Pass the backprojection filter to ProjectionStackToFourD*/
    void SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg);

    /** Pass the forward projection filter to FourDToProjectionStack */
    void SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg);

    /** Pass the geometry to all filters needing it */
    itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

    /** Use CUDA interpolation/splat filters */
    itkSetMacro(UseCudaInterpolation, bool)
    itkGetMacro(UseCudaInterpolation, bool)
    itkSetMacro(UseCudaSplat, bool)
    itkGetMacro(UseCudaSplat, bool)
    itkSetMacro(UseCudaSources, bool)
    itkGetMacro(UseCudaSources, bool)

    /** Macros that take care of implementing the Get and Set methods for Weights.*/
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

    /** Store the phase signal in a member variable */
    virtual void SetSignal(const std::vector<double> signal);

    /** Set / Get whether the displaced detector filter should be disabled */
    itkSetMacro(DisableDisplacedDetectorFilter, bool)
    itkGetMacro(DisableDisplacedDetectorFilter, bool)

protected:
    FourDReconstructionConjugateGradientOperator();
    ~FourDReconstructionConjugateGradientOperator() {}

    /** Builds the pipeline and computes output information */
    void GenerateOutputInformation() ITK_OVERRIDE;

    /** Computes the requested region of input images */
    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    /** Initialize the ConstantImageSourceFilter */
    void InitializeConstantSources();

    /** Member pointers to the filters used internally (for convenience)*/
    typename BackProjectionFilterType::Pointer            m_BackProjectionFilter;
    typename ForwardProjectionFilterType::Pointer         m_ForwardProjectionFilter;
    typename InterpolationFilterType::Pointer             m_InterpolationFilter;
    typename SplatFilterType::Pointer                     m_SplatFilter;
    typename ConstantVolumeSourceType::Pointer            m_ConstantVolumeSource1;
    typename ConstantVolumeSourceType::Pointer            m_ConstantVolumeSource2;
    typename ConstantProjectionStackSourceType::Pointer   m_ConstantProjectionStackSource;
    typename ConstantVolumeSeriesSourceType::Pointer      m_ConstantVolumeSeriesSource;
    typename DisplacedDetectorFilterType::Pointer         m_DisplacedDetectorFilter;

    ThreeDCircularProjectionGeometry::Pointer             m_Geometry;
    bool                                                  m_UseCudaInterpolation;
    bool                                                  m_UseCudaSplat;
    bool                                                  m_UseCudaSources;
    itk::Array2D<float>                                   m_Weights;
    std::vector<double>                                   m_Signal;
    bool                                                  m_DisableDisplacedDetectorFilter;

private:
    FourDReconstructionConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDReconstructionConjugateGradientOperator.hxx"
#endif

#endif
