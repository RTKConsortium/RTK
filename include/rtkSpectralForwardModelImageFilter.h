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

#include "rtkSchlomka2008NegativeLogLikelihood.h"
#include "rtkDualEnergyNegativeLogLikelihood.h"

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
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType = itk::VectorImage<float, 2>,
          typename DetectorResponseImageType = itk::Image<float, 2>,
          typename MaterialAttenuationsImageType = itk::Image<float, 2>>
class ITK_EXPORT SpectralForwardModelImageFilter
  : public itk::InPlaceImageFilter<MeasuredProjectionsType, MeasuredProjectionsType>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(SpectralForwardModelImageFilter);

  /** Standard class type alias. */
  using Self = SpectralForwardModelImageFilter;
  using Superclass = itk::ImageToImageFilter<MeasuredProjectionsType, MeasuredProjectionsType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = MeasuredProjectionsType;
  using OutputImageType = MeasuredProjectionsType;

  /** Convenient information */
  using ThresholdsType = itk::VariableLengthVector<double>;
  using DetectorResponseType = vnl_matrix<double>;
  using MaterialAttenuationsType = vnl_matrix<double>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(SpectralForwardModelImageFilter, InPlaceImageFilter);

  /** Set/Get the incident spectrum input images */
  void
  SetInputIncidentSpectrum(const IncidentSpectrumImageType * IncidentSpectrum);
  void
  SetInputSecondIncidentSpectrum(const IncidentSpectrumImageType * IncidentSpectrum);
  typename IncidentSpectrumImageType::ConstPointer
  GetInputIncidentSpectrum();
  typename IncidentSpectrumImageType::ConstPointer
  GetInputSecondIncidentSpectrum();

  /** Set/Get the input material-decomposed stack of projections (only used for initialization) */
  void
  SetInputDecomposedProjections(const DecomposedProjectionsType * DecomposedProjections);
  typename DecomposedProjectionsType::ConstPointer
  GetInputDecomposedProjections();

  /** Set/Get the input stack of measured projections (to be decomposed in materials) */
  void
  SetInputMeasuredProjections(const MeasuredProjectionsType * SpectralProjections);
  typename MeasuredProjectionsType::ConstPointer
  GetInputMeasuredProjections();

  /** Set/Get the detector response as an image */
  void
  SetDetectorResponse(const DetectorResponseImageType * DetectorResponse);
  typename DetectorResponseImageType::ConstPointer
  GetDetectorResponse();

  /** Set/Get the material attenuations as an image */
  void
  SetMaterialAttenuations(const MaterialAttenuationsImageType * MaterialAttenuations);
  typename MaterialAttenuationsImageType::ConstPointer
  GetMaterialAttenuations();

  itkSetMacro(Thresholds, ThresholdsType);
  itkGetMacro(Thresholds, ThresholdsType);

  itkSetMacro(NumberOfSpectralBins, unsigned int);
  itkGetMacro(NumberOfSpectralBins, unsigned int);

  itkSetMacro(NumberOfMaterials, unsigned int);
  itkGetMacro(NumberOfMaterials, unsigned int);

  itkSetMacro(NumberOfEnergies, unsigned int);
  itkGetMacro(NumberOfEnergies, unsigned int);

  itkSetMacro(IsSpectralCT, bool);
  itkGetMacro(IsSpectralCT, bool);

  itkSetMacro(ComputeVariances, bool);
  itkGetMacro(ComputeVariances, bool);

protected:
  SpectralForwardModelImageFilter();
  ~SpectralForwardModelImageFilter() override = default;

  /**  Create the Output */
  using DataObjectPointerArraySizeType = itk::ProcessObject::DataObjectPointerArraySizeType;
  using Superclass::MakeOutput;
  itk::DataObject::Pointer
  MakeOutput(DataObjectPointerArraySizeType idx) override;

  void
  GenerateOutputInformation() override;

  void
  GenerateInputRequestedRegion() override;

  void
  BeforeThreadedGenerateData() override;
  void
  DynamicThreadedGenerateData(const typename OutputImageType::RegionType & outputRegionForThread) override;

  /** The inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

  ThresholdsType m_Thresholds;
  unsigned int   m_NumberOfSpectralBins;

  MaterialAttenuationsType m_MaterialAttenuations;
  DetectorResponseType     m_DetectorResponse;
  unsigned int             m_NumberOfEnergies;

  /** Parameters */
  unsigned int m_NumberOfIterations;
  unsigned int m_NumberOfMaterials;
  bool         m_OptimizeWithRestarts;
  bool         m_IsSpectralCT;     // If not, it is dual energy CT
  bool         m_ComputeVariances; // Only implemented for dual energy CT

}; // end of class

// Function to bin a detector response matrix according to given energy thresholds
template <typename OutputElementType, typename DetectorResponseImageType, typename ThresholdsType>
vnl_matrix<OutputElementType>
SpectralBinDetectorResponse(const DetectorResponseImageType * drm,
                            const ThresholdsType &            thresholds,
                            const unsigned int                numberOfEnergies);

} // end namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSpectralForwardModelImageFilter.hxx"
#endif

#endif
