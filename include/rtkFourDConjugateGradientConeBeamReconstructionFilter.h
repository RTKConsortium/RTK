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

#ifndef rtkFourDConjugateGradientConeBeamReconstructionFilter_h
#define rtkFourDConjugateGradientConeBeamReconstructionFilter_h

#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkConjugateGradientImageFilter.h"
#include "rtkFourDReconstructionConjugateGradientOperator.h"
#include "rtkProjectionStackToFourDImageFilter.h"
#include "rtkDisplacedDetectorImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkIterationReporter.h>
#ifdef RTK_USE_CUDA
#  include "rtkCudaConjugateGradientImageFilter.h"
#endif

namespace rtk
{

/** \class FourDConjugateGradientConeBeamReconstructionFilter
 * \brief Implements part of the 4D reconstruction by conjugate gradient
 *
 * See the reference paper: "Cardiac C-arm computed tomography using
 * a 3D + time ROI reconstruction method with spatial and temporal regularization"
 * by Mory et al.
 *
 * 4D conjugate gradient reconstruction consists in iteratively
 * minimizing the following cost function:
 *
 * \f[ \sum\limits_{\alpha} \| R_\alpha S_\alpha x - p_\alpha \|_2^2 \f]
 *
 * with
 * - \f$ x \f$ a 4D series of 3D volumes, each one being the reconstruction
 * at a given respiratory/cardiac phase
 * - \f$ p_{\alpha} \f$ is the projection measured at angle \f$ \alpha \f$
 * - \f$ S_{\alpha} \f$ an interpolation operator which, from the 3D + time sequence f,
 * estimates the 3D volume through which projection \f$ p_{\alpha} \f$ has been acquired
 * - \f$ R_{\alpha} \f$ is the X-ray transform (the forward projection operator) for angle \f$ \alpha \f$
 * - \f$ D \f$ the displaced detector weighting matrix
 *
 * \dot
 * digraph FourDConjugateGradientConeBeamReconstructionFilter {
 *
 * Input0 [ label="Input 0 (Input: 4D sequence of volumes)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
 * ConjugateGradient [ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
 * PSTFD [ label="rtk::ProjectionStackToFourDImageFilter" URL="\ref rtk::ProjectionStackToFourDImageFilter"];
 * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
 *
 * Input0 -> AfterInput0 [arrowhead=none];
 * AfterInput0 -> ConjugateGradient;
 * Input0 -> PSTFD;
 * Input1 -> Displaced;
 * Displaced -> PSTFD;
 * PSTFD -> ConjugateGradient;
 * ConjugateGradient -> Output;
 * }
 * \enddot
 *
 * \test rtkfourdconjugategradienttest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename VolumeSeriesType, typename ProjectionStackType>
class ITK_TEMPLATE_EXPORT FourDConjugateGradientConeBeamReconstructionFilter
  : public rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FourDConjugateGradientConeBeamReconstructionFilter);

  /** Standard class type alias. */
  using Self = FourDConjugateGradientConeBeamReconstructionFilter;
  using Superclass = IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = VolumeSeriesType;
  using OutputImageType = VolumeSeriesType;
  using VolumeType = ProjectionStackType;

  /** Typedefs of each subfilter of this composite filter */
  using ForwardProjectionFilterType = ForwardProjectionImageFilter<VolumeType, ProjectionStackType>;
  using BackProjectionFilterType = BackProjectionImageFilter<ProjectionStackType, VolumeType>;
  using ConjugateGradientFilterType = ConjugateGradientImageFilter<VolumeSeriesType>;
  using CGOperatorFilterType = FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>;
  using ProjStackToFourDFilterType = ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>;
  using DisplacedDetectorFilterType = DisplacedDetectorImageFilter<ProjectionStackType>;

  using ForwardProjectionType = typename Superclass::ForwardProjectionType;
  using BackProjectionType = typename Superclass::BackProjectionType;

  /** SFINAE type alias, depending on whether a CUDA image is used. */
  using CPUVolumeSeriesType =
    typename itk::Image<typename VolumeSeriesType::PixelType, VolumeSeriesType::ImageDimension>;
#ifdef RTK_USE_CUDA
  using CudaConjugateGradientImageFilterType =
    typename std::conditional_t<std::is_same_v<VolumeSeriesType, CPUVolumeSeriesType>,
                                ConjugateGradientImageFilter<VolumeSeriesType>,
                                CudaConjugateGradientImageFilter<VolumeSeriesType>>;
#else
  using CudaConjugateGradientImageFilterType = ConjugateGradientImageFilter<VolumeSeriesType>;
#endif

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(FourDConjugateGradientConeBeamReconstructionFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetConstObjectMacro(Geometry, ThreeDCircularProjectionGeometry);
  itkSetConstObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

  /** Get / Set whether conjugate gradient should be performed on GPU */
  itkGetMacro(CudaConjugateGradient, bool);
  itkSetMacro(CudaConjugateGradient, bool);

  /** Set/Get the 4D image to be updated.*/
  void
  SetInputVolumeSeries(const VolumeSeriesType * VolumeSeries);
  typename VolumeSeriesType::ConstPointer
  GetInputVolumeSeries();

  /** Set/Get the stack of projections */
  void
  SetInputProjectionStack(const ProjectionStackType * Projections);
  typename ProjectionStackType::ConstPointer
  GetInputProjectionStack();

  /** Pass the interpolation weights to subfilters */
  void
  SetWeights(const itk::Array2D<float> _arg);

  /** Store the phase signal in a member variable */
  virtual void
  SetSignal(const std::vector<double> signal);

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool);
  itkGetMacro(DisableDisplacedDetectorFilter, bool);

protected:
  FourDConjugateGradientConeBeamReconstructionFilter();
  ~FourDConjugateGradientConeBeamReconstructionFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() const override;

  void
  GenerateOutputInformation() override;

  void
  GenerateInputRequestedRegion() override;

  void
  GenerateData() override;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

  /** Pointers to each subfilter of this composite filter */
  typename ForwardProjectionFilterType::Pointer m_ForwardProjectionFilter;
  typename BackProjectionFilterType::Pointer    m_BackProjectionFilter;
  typename BackProjectionFilterType::Pointer    m_BackProjectionFilterForB;
  typename ConjugateGradientFilterType::Pointer m_ConjugateGradientFilter;
  typename CGOperatorFilterType::Pointer        m_CGOperator;
  typename ProjStackToFourDFilterType::Pointer  m_ProjStackToFourDFilter;
  typename DisplacedDetectorFilterType::Pointer m_DisplacedDetectorFilter;

  bool                m_CudaConjugateGradient;
  std::vector<double> m_Signal;
  bool                m_DisableDisplacedDetectorFilter;

  // Iteration reporting
  itk::IterationReporter m_IterationReporter;

private:
  /** Geometry object */
  ThreeDCircularProjectionGeometry::ConstPointer m_Geometry;

  /** Number of conjugate gradient descent iterations */
  unsigned int m_NumberOfIterations;

  /** Iteration reporter */
  void
  ReportProgress(itk::Object *, const itk::EventObject &);

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFourDConjugateGradientConeBeamReconstructionFilter.hxx"
#endif

#endif
