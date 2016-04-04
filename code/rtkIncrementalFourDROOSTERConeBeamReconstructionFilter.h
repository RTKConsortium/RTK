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

#ifndef __rtkIncrementalFourDROOSTERConeBeamReconstructionFilter_h
#define __rtkIncrementalFourDROOSTERConeBeamReconstructionFilter_h

#include "rtkPhasesToInterpolationWeights.h"
#include "rtkFourDROOSTERConeBeamReconstructionFilter.h"
#include "rtkSubSelectFromListImageFilter.h"

namespace rtk
{

  /** \class IncrementalFourDROOSTERConeBeamReconstructionFilter
   * \brief Implements an incremental version of 4D ROOSTER, described in Mory, Rit, Sixou
   * "4D Tomography: an Application of Incremental Constraint Projection Methods for 
   * Variational Inequalities", Proceedings of GRETSI 2015
   *
   * \test rtkIncrementalFourDROOSTERtest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template<typename VolumeSeriesType, typename ProjectionStackType>
class ITK_EXPORT IncrementalFourDROOSTERConeBeamReconstructionFilter :
  public rtk::FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
  /** Standard class typedefs. */
  typedef IncrementalFourDROOSTERConeBeamReconstructionFilter                               Self;
  typedef FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>   Superclass;
  typedef itk::SmartPointer<Self>                                                           Pointer;
  typedef itk::SmartPointer<const Self>                                                     ConstPointer;

  /** Some convenient typedefs. */
  typedef VolumeSeriesType      InputImageType;
  typedef VolumeSeriesType      OutputImageType;
  typedef ProjectionStackType   VolumeType;

  /** Typedefs of each subfilter of this composite filter */
  typedef rtk::SubSelectFromListImageFilter<ProjectionStackType>    SubSelectType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(IncrementalFourDROOSTERConeBeamReconstructionFilter, FourDROOSTERConeBeamReconstructionFilter)

  /** Set the name of the file containing the phase of each projection */
  itkSetStringMacro(PhasesFileName)

  /** Get / Set the number of projections per subset. Default is all (1 subset) */
  itkGetMacro(NumberOfProjectionsPerSubset, unsigned int)
  itkSetMacro(NumberOfProjectionsPerSubset, unsigned int)

  /** Set the Kzero parameter used to calculate alpha_k */
  itkGetMacro(Kzero, float)
  itkSetMacro(Kzero, float)

  /** Store the phase signal in a member variable */
  virtual void SetSignal(const std::vector<double> signal);

protected:
  IncrementalFourDROOSTERConeBeamReconstructionFilter();
  ~IncrementalFourDROOSTERConeBeamReconstructionFilter(){}

  virtual void GenerateOutputInformation();

  virtual void GenerateData();

  // Override SetWeights from the mother class to do nothing
  // Here the interpolation weights for the 4D CG filter depend
  // on the subset of projections being processed
  virtual void SetWeights(const itk::Array2D<float> _arg){}

  /** Pointers to each subfilter of this composite filter */
  typename std::vector<typename SubSelectType::Pointer>                   m_SubSelectFilters;
  typename std::vector<typename PhasesToInterpolationWeights::Pointer>    m_PhasesFilters;

  /** A vector containing as many vectors of booleans as the number of subsets. Each boolean
  indicates whether the corresponding projection is selected in the subset */
  std::vector< std::vector< bool > > m_SelectedProjsVector;

  /** Number of projections in each subset (given by user), and number of subsets (computed) */
  unsigned int m_NumberOfProjectionsPerSubset;
  unsigned int m_NumberOfSubsets;

  /** Name of the file containing the phases */
  std::string             m_PhasesFileName;
  std::vector<double>     m_Signal;

  /** Convergence parameter */
  float m_Kzero;

private:
  //purposely not implemented
  IncrementalFourDROOSTERConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkIncrementalFourDROOSTERConeBeamReconstructionFilter.hxx"
#endif

#endif
