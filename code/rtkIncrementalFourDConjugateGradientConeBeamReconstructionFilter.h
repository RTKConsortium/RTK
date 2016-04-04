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

#ifndef __rtkIncrementalFourDConjugateGradientConeBeamReconstructionFilter_h
#define __rtkIncrementalFourDConjugateGradientConeBeamReconstructionFilter_h

#include "rtkPhasesToInterpolationWeights.h"
#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkSubSelectFromListImageFilter.h"

#include <itkTimeProbe.h>

namespace rtk
{

  /** \class IncrementalFourDConjugateGradientConeBeamReconstructionFilter
   * \brief Implements incremental 4D reconstruction by conjugate gradient
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
   * This filter splits the cost function into smaller terms, each one
   * representing the attachment to a subset of the projections. They are minimized
   * in an alternating fashion.
   *
   * \dot
   * digraph IncrementalFourDConjugateGradientConeBeamReconstructionFilter {
   *
   * Input0 [ label="Input 0 (Input: 4D sequence of volumes)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * AfterInput1 [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterCG [label="", fixedsize="false", width=0, height=0, shape=none];
   * BeforeCG [label="", fixedsize="false", width=0, height=0, shape=none];
   * BeforeCGVol [label="", fixedsize="false", width=0, height=0, shape=none];
   * BeforeCGWeights [label="", fixedsize="false", width=0, height=0, shape=none];
   * ConjugateGradient [ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
   * SubSelect1 [label="rtk::SubSelectFromListImageFilter" URL="\ref rtk::SubSelectFromListImageFilter"];
   * SubSelect2 [label="rtk::SubSelectFromListImageFilter" URL="\ref rtk::SubSelectFromListImageFilter"];
   * SubSelect3 [label="rtk::SubSelectFromListImageFilter" URL="\ref rtk::SubSelectFromListImageFilter"];
   * InterpWeights1 [label="rtk::PhasesToInterpolationWeights" URL="\ref rtk::PhasesToInterpolationWeights"];
   * InterpWeights2 [label="rtk::PhasesToInterpolationWeights" URL="\ref rtk::PhasesToInterpolationWeights"];
   * InterpWeights3 [label="rtk::PhasesToInterpolationWeights" URL="\ref rtk::PhasesToInterpolationWeights"];
   *
   * Input0 -> BeforeCGVol[arrowhead=none];
   * BeforeCGVol -> ConjugateGradient;
   * Input1 -> AfterInput1[arrowhead=none];
   * AfterInput1 -> SubSelect1;
   * AfterInput1 -> InterpWeights1;
   * AfterInput1 -> SubSelect2;
   * AfterInput1 -> InterpWeights2;
   * AfterInput1 -> SubSelect3;
   * AfterInput1 -> InterpWeights3;
   * SubSelect1 -> BeforeCG[arrowhead=none];
   * SubSelect2 -> BeforeCG[arrowhead=none];
   * SubSelect3 -> BeforeCG[arrowhead=none];
   * InterpWeights1 -> BeforeCGWeights[arrowhead=none];
   * InterpWeights2 -> BeforeCGWeights[arrowhead=none];
   * InterpWeights3 -> BeforeCGWeights[arrowhead=none];
   * BeforeCG -> ConjugateGradient;
   * BeforeCGWeights -> ConjugateGradient;
   * ConjugateGradient -> AfterCG;
   * AfterCG -> BeforeCGVol[style=dashed];
   * AfterCG -> Output;
   * }
   * \enddot
   *
   * \test rtkincrementalfourdconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template<typename VolumeSeriesType, typename ProjectionStackType>
class ITK_EXPORT IncrementalFourDConjugateGradientConeBeamReconstructionFilter :
  public rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
  /** Standard class typedefs. */
  typedef IncrementalFourDConjugateGradientConeBeamReconstructionFilter                   Self;
  typedef IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>     Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef VolumeSeriesType      InputImageType;
  typedef VolumeSeriesType      OutputImageType;
  typedef ProjectionStackType   VolumeType;

  /** Typedefs of each subfilter of this composite filter */
  typedef rtk::SubSelectFromListImageFilter<ProjectionStackType>                                          SubSelectType;
  typedef rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>  CGType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(IncrementalFourDConjugateGradientConeBeamReconstructionFilter, IterativeConeBeamReconstructionFilter)

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

  /** Get / Set the number of iterations */
  itkGetMacro(MainLoop_iterations, unsigned int)
  itkSetMacro(MainLoop_iterations, unsigned int)

  /** Get / Set the number of iterations */
  itkGetMacro(CG_iterations, unsigned int)
  itkSetMacro(CG_iterations, unsigned int)

  /** Set/Get the 4D image to be updated.*/
  void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);
  typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();

  /** Set/Get the stack of projections  */
  void SetInputProjectionStack(const VolumeType* Projection);
  typename VolumeType::ConstPointer GetInputProjectionStack();

  /** Pass the ForwardProjection filter to the conjugate gradient operator */
  void SetForwardProjectionFilter (int _arg);

  /** Pass the backprojection filter to the conjugate gradient operator and to the filter generating the B of AX=B */
  void SetBackProjectionFilter (int _arg);

  /** Set the name of the file containing the phase of each projection */
  itkSetStringMacro(PhasesFileName)

  /** Get / Set the number of projections per subset. Default is all (1 subset) */
  itkGetMacro(NumberOfProjectionsPerSubset, unsigned int)
  itkSetMacro(NumberOfProjectionsPerSubset, unsigned int)

  /** Get / Set whether conjugate gradient should be performed on GPU */
  itkGetMacro(CudaConjugateGradient, bool)
  itkSetMacro(CudaConjugateGradient, bool)

  /** Store the phase signal in a member variable */
  virtual void SetSignal(const std::vector<double> signal);

protected:
  IncrementalFourDConjugateGradientConeBeamReconstructionFilter();
  ~IncrementalFourDConjugateGradientConeBeamReconstructionFilter(){}

  virtual void GenerateOutputInformation();

  virtual void GenerateData();

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

  /** Pointers to each subfilter of this composite filter */
  typename std::vector<typename SubSelectType::Pointer>                   m_SubSelectFilters;
  typename std::vector<typename PhasesToInterpolationWeights::Pointer>    m_PhasesFilters;
  typename CGType::Pointer                                                m_CG;

  /** A vector containing as many vectors of booleans as the number of subsets. Each boolean
  indicates whether the corresponding projection is selected in the subset */
  std::vector< std::vector< bool > > m_SelectedProjsVector;

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of iterations */
  unsigned int m_MainLoop_iterations; //Main loop
  unsigned int m_CG_iterations; //Conjugate gradient iterations each time a subset is processed

  /** Number of projections in each subset (given by user), and number of subsets (computed) */
  unsigned int m_NumberOfProjectionsPerSubset;
  unsigned int m_NumberOfSubsets;

  /** Name of the file containing the phases */
  std::string             m_PhasesFileName;
  std::vector<double>     m_Signal;

  /** Should conjugate gradient be performed on GPU ? */
  bool  m_CudaConjugateGradient;

private:
  //purposely not implemented
  IncrementalFourDConjugateGradientConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkIncrementalFourDConjugateGradientConeBeamReconstructionFilter.hxx"
#endif

#endif
