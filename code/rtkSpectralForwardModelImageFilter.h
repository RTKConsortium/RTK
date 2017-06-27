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

#ifndef rtkSpectralForwardModelImageFilter_h
#define rtkSpectralForwardModelImageFilter_h

#include "rtkSimplexProjectionsDecompositionImageFilter.h"
#include "rtkSchlomka2008NegativeLogLikelihood.h"

#include <itkInPlaceImageFilter.h>

namespace rtk
{
  /** \class SpectralForwardModelImageFilter
   * \brief Forward model for the decomposition of spectral projection images into material projections.
   *
   * Computes the photon counts from the decomposed projections, the incident spectrum,
   * the detector response and the material attenuation curves.
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
class ITK_EXPORT SpectralForwardModelImageFilter :
  public itk::InPlaceImageFilter<MeasuredProjectionsType, MeasuredProjectionsType>
{
public:
  /** Standard class typedefs. */
  typedef SpectralForwardModelImageFilter                                                Self;
  typedef itk::ImageToImageFilter<MeasuredProjectionsType, MeasuredProjectionsType>      Superclass;
  typedef itk::SmartPointer<Self>                                                        Pointer;
  typedef itk::SmartPointer<const Self>                                                  ConstPointer;

  /** Some convenient typedefs. */
  typedef MeasuredProjectionsType       InputImageType;
  typedef MeasuredProjectionsType       OutputImageType;

  /** Convenient information */
  typedef itk::VariableLengthVector<unsigned int>           ThresholdsType;
  typedef itk::VariableSizeMatrix<double>                   DetectorResponseType;
  typedef itk::VariableSizeMatrix<double>                   MaterialAttenuationsType;

  /** Typedefs of each subfilter of this composite filter */
  typedef Schlomka2008NegativeLogLikelihood                             CostFunctionType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(SpectralForwardModelImageFilter, InPlaceImageFilter)

  /** Set/Get the incident spectrum input image */
  void SetInputIncidentSpectrum(const IncidentSpectrumImageType* IncidentSpectrum);
  typename IncidentSpectrumImageType::ConstPointer GetInputIncidentSpectrum();

  /** Set/Get the input material-decomposed stack of projections (only used for initialization) */
  void SetInputDecomposedProjections(const DecomposedProjectionsType* DecomposedProjections);
  typename DecomposedProjectionsType::ConstPointer GetInputDecomposedProjections();

  /** Set/Get the input stack of measured projections (to be decomposed in materials) */
  void SetInputMeasuredProjections(const MeasuredProjectionsType* SpectralProjections);
  typename MeasuredProjectionsType::ConstPointer GetInputMeasuredProjections();

  /** Set/Get the detector response as an image */
  void SetDetectorResponse(const DetectorResponseImageType* DetectorResponse);
  typename DetectorResponseImageType::ConstPointer GetDetectorResponse();

  /** Set/Get the material attenuations as an image */
  void SetMaterialAttenuations(const MaterialAttenuationsImageType* MaterialAttenuations);
  typename MaterialAttenuationsImageType::ConstPointer GetMaterialAttenuations();

  itkSetMacro(Thresholds, ThresholdsType)
  itkGetMacro(Thresholds, ThresholdsType)

  itkSetMacro(NumberOfSpectralBins, unsigned int)
  itkGetMacro(NumberOfSpectralBins, unsigned int)

  itkSetMacro(NumberOfMaterials, unsigned int)
  itkGetMacro(NumberOfMaterials, unsigned int)

  itkSetMacro(NumberOfEnergies, unsigned int)
  itkGetMacro(NumberOfEnergies, unsigned int)

protected:
  SpectralForwardModelImageFilter();
  ~SpectralForwardModelImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;
  void ThreadedGenerateData(const typename OutputImageType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;

  /** The inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  ThresholdsType             m_Thresholds;
  unsigned int               m_NumberOfSpectralBins;

  MaterialAttenuationsType   m_MaterialAttenuations;
  DetectorResponseType       m_DetectorResponse;
  unsigned int               m_NumberOfEnergies;

  /** Parameters */
  unsigned int m_NumberOfIterations;
  unsigned int m_NumberOfMaterials;
  bool         m_OptimizeWithRestarts;

private:
  //purposely not implemented
  SpectralForwardModelImageFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSpectralForwardModelImageFilter.hxx"
#endif

#endif
