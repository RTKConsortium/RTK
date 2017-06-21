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

#ifndef rtkSimplexSpectralProjectionsDecompositionImageFilter_h
#define rtkSimplexSpectralProjectionsDecompositionImageFilter_h

#include "rtkSimplexProjectionsDecompositionImageFilter.h"
#include "rtkSchlomka2008NegativeLogLikelihood.h"

namespace rtk
{
  /** \class SimplexSpectralProjectionsDecompositionImageFilter
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
class ITK_EXPORT SimplexSpectralProjectionsDecompositionImageFilter :
  public rtk::SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, DecomposedProjectionsType>
{
public:
  /** Standard class typedefs. */
  typedef SimplexSpectralProjectionsDecompositionImageFilter                             Self;
  typedef rtk::SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                          DecomposedProjectionsType>     Superclass;
  typedef itk::SmartPointer<Self>                                                        Pointer;
  typedef itk::SmartPointer<const Self>                                                  ConstPointer;

  /** Some convenient typedefs. */
  typedef DecomposedProjectionsType       InputImageType;
  typedef DecomposedProjectionsType       OutputImageType;

  /** Convenient information */
  typedef itk::VariableLengthVector<unsigned int>   ThresholdsType;
  typedef itk::VariableSizeMatrix<double>           MeanAttenuationInBinType;

  /** Typedefs of each subfilter of this composite filter */
  typedef Schlomka2008NegativeLogLikelihood                             CostFunctionType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(SimplexSpectralProjectionsDecompositionImageFilter, SimplexProjectionsDecompositionImageFilter)

  /** Set/Get the incident spectrum input image */
  void SetInputIncidentSpectrum(const IncidentSpectrumImageType* IncidentSpectrum);
  typename IncidentSpectrumImageType::ConstPointer GetInputIncidentSpectrum();

  itkSetMacro(Thresholds, ThresholdsType)
  itkGetMacro(Thresholds, ThresholdsType)

  itkSetMacro(NumberOfSpectralBins, unsigned int)
  itkGetMacro(NumberOfSpectralBins, unsigned int)

  itkSetMacro(OutputInverseCramerRaoLowerBound, bool)
  itkGetMacro(OutputInverseCramerRaoLowerBound, bool)

  itkSetMacro(OutputFischerMatrix, bool)
  itkGetMacro(OutputFischerMatrix, bool)

  itkSetMacro(LogTransformEachBin, bool)
  itkGetMacro(LogTransformEachBin, bool)

  itkSetMacro(GuessInitialization, bool)
  itkGetMacro(GuessInitialization, bool)

protected:
  SimplexSpectralProjectionsDecompositionImageFilter();
  ~SimplexSpectralProjectionsDecompositionImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;
  void ThreadedGenerateData(const typename DecomposedProjectionsType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;

  /** The inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  ThresholdsType             m_Thresholds;
  unsigned int               m_NumberOfSpectralBins;
  bool                       m_OutputInverseCramerRaoLowerBound;
  bool                       m_OutputFischerMatrix;
  bool                       m_LogTransformEachBin;
  bool                       m_GuessInitialization;
  MeanAttenuationInBinType   m_MeanAttenuationInBin;

private:
  //purposely not implemented
  SimplexSpectralProjectionsDecompositionImageFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.hxx"
#endif

#endif
