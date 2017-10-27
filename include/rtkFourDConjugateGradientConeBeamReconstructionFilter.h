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
#include <itkTimeProbe.h>
#ifdef RTK_USE_CUDA
  #include "rtkCudaConjugateGradientImageFilter_4f.h"
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
   * \ingroup ReconstructionAlgorithm
   */

template<typename VolumeSeriesType, typename ProjectionStackType>
class ITK_EXPORT FourDConjugateGradientConeBeamReconstructionFilter :
  public rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
  /** Standard class typedefs. */
  typedef FourDConjugateGradientConeBeamReconstructionFilter                   Self;
  typedef IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>     Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef VolumeSeriesType      InputImageType;
  typedef VolumeSeriesType      OutputImageType;
  typedef ProjectionStackType   VolumeType;

  /** Typedefs of each subfilter of this composite filter */
  typedef rtk::ForwardProjectionImageFilter< VolumeType, ProjectionStackType >                      ForwardProjectionFilterType;
  typedef rtk::BackProjectionImageFilter< ProjectionStackType, VolumeType >                         BackProjectionFilterType;
  typedef rtk::ConjugateGradientImageFilter<VolumeSeriesType>                                       ConjugateGradientFilterType;
  typedef rtk::FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>  CGOperatorFilterType;
  typedef rtk::ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>             ProjStackToFourDFilterType;
  typedef rtk::DisplacedDetectorImageFilter<ProjectionStackType>                                    DisplacedDetectorFilterType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(FourDConjugateGradientConeBeamReconstructionFilter, itk::ImageToImageFilter)

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

  void PrintTiming(std::ostream& os) const;

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(NumberOfIterations, unsigned int)
  itkSetMacro(NumberOfIterations, unsigned int)

  /** Get / Set whether conjugate gradient should be performed on GPU */
  itkGetMacro(CudaConjugateGradient, bool)
  itkSetMacro(CudaConjugateGradient, bool)

  /** Set/Get the 4D image to be updated.*/
  void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);
  typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();

  /** Set/Get the stack of projections */
  void SetInputProjectionStack(const ProjectionStackType* Projections);
  typename ProjectionStackType::ConstPointer GetInputProjectionStack();

  /** Pass the ForwardProjection filter to the conjugate gradient operator */
  void SetForwardProjectionFilter (int _arg) ITK_OVERRIDE;

  /** Pass the backprojection filter to the conjugate gradient operator and to the filter generating the B of AX=B */
  void SetBackProjectionFilter (int _arg) ITK_OVERRIDE;

  /** Pass the interpolation weights to subfilters */
  void SetWeights(const itk::Array2D<float> _arg);

  /** Store the phase signal in a member variable */
  virtual void SetSignal(const std::vector<double> signal);

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool)
  itkGetMacro(DisableDisplacedDetectorFilter, bool)
protected:
  FourDConjugateGradientConeBeamReconstructionFilter();
  ~FourDConjugateGradientConeBeamReconstructionFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void GenerateData() ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  /** Pointers to each subfilter of this composite filter */
  typename ForwardProjectionFilterType::Pointer     m_ForwardProjectionFilter;
  typename BackProjectionFilterType::Pointer        m_BackProjectionFilter;
  typename BackProjectionFilterType::Pointer        m_BackProjectionFilterForB;
  typename ConjugateGradientFilterType::Pointer     m_ConjugateGradientFilter;
  typename CGOperatorFilterType::Pointer            m_CGOperator;
  typename ProjStackToFourDFilterType::Pointer      m_ProjStackToFourDFilter;
  typename DisplacedDetectorFilterType::Pointer     m_DisplacedDetectorFilter;

  bool                    m_CudaConjugateGradient;
  std::vector<double>     m_Signal;
  bool                    m_DisableDisplacedDetectorFilter;

private:
  //purposely not implemented
  FourDConjugateGradientConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of conjugate gradient descent iterations */
  unsigned int m_NumberOfIterations;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.hxx"
#endif

#endif
