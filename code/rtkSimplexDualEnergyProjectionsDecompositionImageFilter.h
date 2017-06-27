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

#ifndef rtkSimplexDualEnergyProjectionsDecompositionImageFilter_h
#define rtkSimplexDualEnergyProjectionsDecompositionImageFilter_h

#include "rtkSimplexProjectionsDecompositionImageFilter.h"
#include "rtkAlvarez1976NegativeLogLikelihood.h"

namespace rtk
{
  /** \class SimplexDualEnergyProjectionsDecompositionImageFilter
   * \brief Decomposition of spectral projection images into material projections
   *
   * See the reference paper: "Experimental feasibility of multi-energy photon-counting
   * K-edge imaging in pre-clinical computed tomography", Schlomka et al, PMB 2008
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template<typename DecomposedProjectionsType,
         typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType = itk::VectorImage<float, 2>,
         typename DetectorResponseImageType = itk::Image<float, 2>,
         typename MaterialAttenuationsImageType = itk::Image<float, 2> >
class ITK_EXPORT SimplexDualEnergyProjectionsDecompositionImageFilter :
  public rtk::SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                         MeasuredProjectionsType,
                                                         IncidentSpectrumImageType,
                                                         DetectorResponseImageType,
                                                         MaterialAttenuationsImageType>
{
public:
  /** Standard class typedefs. */
  typedef SimplexDualEnergyProjectionsDecompositionImageFilter                           Self;
  typedef rtk::SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                          MeasuredProjectionsType,
                                                          IncidentSpectrumImageType,
                                                          DetectorResponseImageType,
                                                          MaterialAttenuationsImageType> Superclass;
  typedef itk::SmartPointer<Self>                                                        Pointer;
  typedef itk::SmartPointer<const Self>                                                  ConstPointer;

  /** Some convenient typedefs. */
  typedef DecomposedProjectionsType               InputImageType;
  typedef DecomposedProjectionsType               OutputImageType;

  /** Typedefs of each subfilter of this composite filter */
  typedef Alvarez1976NegativeLogLikelihood        CostFunctionType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(SimplexDualEnergyProjectionsDecompositionImageFilter, SimplexProjectionsDecompositionImageFilter)

  /** Set/Get the high energy incident spectrum input image */
  void SetHighEnergyIncidentSpectrum(const IncidentSpectrumImageType* IncidentSpectrum);
  typename IncidentSpectrumImageType::ConstPointer GetHighEnergyIncidentSpectrum();

  /** Set/Get the low energy incident spectrum input image */
  void SetLowEnergyIncidentSpectrum(const IncidentSpectrumImageType* IncidentSpectrum);
  typename IncidentSpectrumImageType::ConstPointer GetLowEnergyIncidentSpectrum();

protected:
  SimplexDualEnergyProjectionsDecompositionImageFilter();
  ~SimplexDualEnergyProjectionsDecompositionImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;
  void ThreadedGenerateData(const typename DecomposedProjectionsType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;

  /** The inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

private:
  //purposely not implemented
  SimplexDualEnergyProjectionsDecompositionImageFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSimplexDualEnergyProjectionsDecompositionImageFilter.hxx"
#endif

#endif
