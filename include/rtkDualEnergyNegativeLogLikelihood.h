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

#ifndef rtkDualEnergyNegativeLogLikelihood_h
#define rtkDualEnergyNegativeLogLikelihood_h

#include "rtkProjectionsDecompositionNegativeLogLikelihood.h"
#include "rtkMacro.h"

#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <itkVariableSizeMatrix.h>

namespace rtk
{
/** \class rtkDualEnergyNegativeLogLikelihood
 * \brief Cost function for dual energy decomposition into material, and associated forward model
 *
 * This class requires the method "Initialize()" to be run once, before it
 * is passed to the simplex minimizer
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

// We have to define the cost function first
class DualEnergyNegativeLogLikelihood : public rtk::ProjectionsDecompositionNegativeLogLikelihood
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(DualEnergyNegativeLogLikelihood);
#else
  ITK_DISALLOW_COPY_AND_MOVE(DualEnergyNegativeLogLikelihood);
#endif

  using Self = DualEnergyNegativeLogLikelihood;
  using Superclass = rtk::ProjectionsDecompositionNegativeLogLikelihood;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  itkNewMacro(Self);
  itkTypeMacro(DualEnergyNegativeLogLikelihood, rtk::ProjectionsDecompositionNegativeLogLikelihood);

  using ParametersType = Superclass::ParametersType;
  using DerivativeType = Superclass::DerivativeType;
  using MeasureType = Superclass::MeasureType;

  using DetectorResponseType = Superclass::DetectorResponseType;
  using MaterialAttenuationsType = Superclass::MaterialAttenuationsType;
  using MeasuredDataType = Superclass::MeasuredDataType;
  using IncidentSpectrumType = Superclass::IncidentSpectrumType;

  // Constructor
  DualEnergyNegativeLogLikelihood() { m_NumberOfSpectralBins = 2; }

  // Destructor
  ~DualEnergyNegativeLogLikelihood() override = default;

  void
  Initialize() override
  {
    // This method computes the combined m_IncidentSpectrumAndDetectorResponseProduct
    // from m_DetectorResponse and m_IncidentSpectrum
    m_Thresholds.SetSize(2);
    m_Thresholds[0] = 1;
    m_Thresholds[1] = m_NumberOfEnergies;

    // In dual energy CT, one possible design is to illuminate the object with
    // either a low energy or a high energy spectrum, alternating between the two. In that case
    // m_DetectorResponse has only one row (there is a single detector) and m_IncidentSpectrum
    // has two rows (one for high energy, the other for low)
    m_IncidentSpectrumAndDetectorResponseProduct.set_size(2, m_DetectorResponse.cols());
    for (unsigned int i = 0; i < 2; i++)
      for (unsigned int j = 0; j < m_DetectorResponse.cols(); j++)
        m_IncidentSpectrumAndDetectorResponseProduct[i][j] = m_DetectorResponse[0][j] * m_IncidentSpectrum[i][j];
  }

  // Not used with a simplex optimizer, but may be useful later
  // for gradient based methods
  void
  GetDerivative(const ParametersType & itkNotUsed(lineIntegrals),
                DerivativeType &       itkNotUsed(derivatives)) const override
  {
    itkExceptionMacro(<< "Not implemented");
  }

  // Main method
  MeasureType
  GetValue(const ParametersType & parameters) const override
  {
    // Forward model: compute the expected total energy measured by the detector for each spectrum
    vnl_vector<double> forward = ForwardModel(parameters);
    vnl_vector<double> variances = GetVariances(parameters);

    long double measure = 0;
    // TODO: Improve this estimation
    // We assume that the variance of the integrated energy is equal to the mean
    // From equation (5) of "Cramer-Rao lower bound of basis image noise in multiple-energy x-ray imaging",
    // PMB 2009, Roessl et al, we replace the variance with the mean

    // Compute the negative log likelihood from the expectedEnergies
    for (unsigned int i = 0; i < this->m_NumberOfMaterials; i++)
      measure += std::log((long double)variances[i]) +
                 (forward[i] - this->m_MeasuredData[i]) * (forward[i] - this->m_MeasuredData[i]) / variances[i];
    measure *= 0.5;

    return measure;
  }

  vnl_vector<double>
  GetVariances(const ParametersType & lineIntegrals) const override
  {
    vnl_vector<double> attenuationFactors;
    attenuationFactors.set_size(m_NumberOfEnergies);
    GetAttenuationFactors(lineIntegrals, attenuationFactors);

    // Apply detector response, getting the lambdas
    vnl_vector<double> intermediate;
    intermediate.set_size(m_NumberOfEnergies);
    for (unsigned int i = 0; i < m_NumberOfEnergies; i++)
      intermediate[i] = i + 1;
    intermediate = element_product(attenuationFactors, intermediate);
    return (m_IncidentSpectrumAndDetectorResponseProduct * intermediate);
  }

protected:
  itk::VariableSizeMatrix<float> m_Fischer;
};

} // namespace rtk

#endif
