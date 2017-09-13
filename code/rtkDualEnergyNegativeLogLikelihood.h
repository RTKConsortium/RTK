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

#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <itkVariableSizeMatrix.h>

namespace rtk
{
  /** \class rtkDualEnergyNegativeLogLikelihood
   * \brief Cost function for dual energy simplex decomposition
   *
   * This implementation assumes that the input spectrum and the
   * detector response are merged into a single 2D + energy input
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

// We have to define the cost function first
class DualEnergyNegativeLogLikelihood : public rtk::ProjectionsDecompositionNegativeLogLikelihood
{
public:

  typedef DualEnergyNegativeLogLikelihood                      Self;
  typedef rtk::ProjectionsDecompositionNegativeLogLikelihood    Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;
  itkNewMacro( Self );
  itkTypeMacro( DualEnergyNegativeLogLikelihood, rtk::ProjectionsDecompositionNegativeLogLikelihood );

  typedef Superclass::ParametersType          ParametersType;
  typedef Superclass::DerivativeType          DerivativeType;
  typedef Superclass::MeasureType             MeasureType;

  typedef Superclass::MaterialAttenuationsType  MaterialAttenuationsType;
  typedef Superclass::MeasuredDataType          MeasuredDataType;
  typedef Superclass::IncidentSpectrumType      SpectrumAndDetectorResponseType;

  // Constructor
  DualEnergyNegativeLogLikelihood()
  {
  this->m_NumberOfMaterials = 2;
  }

  // Destructor
  ~DualEnergyNegativeLogLikelihood()
  {
  }

  vnl_vector<double> ForwardModel(const ParametersType & lineIntegrals) const ITK_OVERRIDE
  {
  // Variable length vector and variable size matrix cannot be used in linear algebra operations
  // Get their vnl counterparts, which can
  vnl_vector<double> vnl_vec_he(GetAttenuatedIncidentSpectrum(m_HighEnergyIncidentSpectrum, lineIntegrals).GetDataPointer(),
                               GetAttenuatedIncidentSpectrum(m_HighEnergyIncidentSpectrum, lineIntegrals).GetSize());
  vnl_vector<double> vnl_vec_le(GetAttenuatedIncidentSpectrum(m_LowEnergyIncidentSpectrum, lineIntegrals).GetDataPointer(),
                               GetAttenuatedIncidentSpectrum(m_LowEnergyIncidentSpectrum, lineIntegrals).GetSize());

  // Detector response is already included in incident spectra
  // Just get the sum of all energies (dual energy detector integrates all incoming energy)
  vnl_vector<double> expectedEnergies;
  expectedEnergies.set_size(this->m_NumberOfMaterials);
  expectedEnergies[0] = vnl_vec_he.sum(); // There should only be one term, the sum is just for type correctness
  expectedEnergies[1] = vnl_vec_le.sum();
  return expectedEnergies;
  }

  itk::VariableLengthVector<double> GetAttenuatedIncidentSpectrum(IncidentSpectrumType incident, const ParametersType & lineIntegrals) const
  {
  // Solid angle of detector pixel, exposure time and mAs should already be
  // taken into account in the incident spectrum image

  if(this->m_NumberOfEnergies != incident.GetSize())
      itkGenericExceptionMacro(<< "Incident spectrum does not have the correct size")

  // Apply attenuation at each energy
  itk::VariableLengthVector<double> attenuatedIncidentSpectrum;
  attenuatedIncidentSpectrum.SetSize(this->m_NumberOfEnergies);
  attenuatedIncidentSpectrum.Fill(0);
  for (unsigned int e=0; e<this->m_NumberOfEnergies; e++)
    {
    double totalAttenuation = 0.;
    for (unsigned int m=0; m<2; m++)
      {
      totalAttenuation += lineIntegrals[m] * m_MaterialAttenuations[m][e];
      }

    attenuatedIncidentSpectrum[e] = incident[e] * std::exp(-totalAttenuation);
    }

  return attenuatedIncidentSpectrum;
  }

  // Main method
  MeasureType  GetValue( const ParametersType & parameters ) const ITK_OVERRIDE
  {
  // Forward model: compute the m_NumberOfEnergiesexpectedEnergies integrated by the detector
  vnl_vector<double> expectedEnergies = ForwardModel(parameters);

  // TODO: Improve this estimation
  // We assume that the variance of the integrated energy is equal to the mean
  // From equation (5) of "Cramér–Rao lower bound of basis image noise in multiple-energy x-ray imaging", PMB 2009, Roessl et al,
  // we replace the variance with the mean

  // Compute the negative log likelihood from the expectedEnergies
  long double measure = 0;
  for (unsigned int i=0; i<this->m_NumberOfMaterials; i++)
    measure += std::log((long double)expectedEnergies[i]) + (expectedEnergies[i] - this->m_MeasuredData[i]) * (expectedEnergies[i] - this->m_MeasuredData[i]) / expectedEnergies[i];
  return (0.5 * measure);
  }

  // Not used with a simplex optimizer, but may be useful later
  // for gradient based methods
  void GetDerivative( const ParametersType & lineIntegrals,
                      DerivativeType & derivatives ) const ITK_OVERRIDE
  {
  }

  itkSetMacro(HighEnergyIncidentSpectrum, IncidentSpectrumType)
  itkGetMacro(HighEnergyIncidentSpectrum, IncidentSpectrumType)

  itkSetMacro(LowEnergyIncidentSpectrum, IncidentSpectrumType)
  itkGetMacro(LowEnergyIncidentSpectrum, IncidentSpectrumType)

protected:
  IncidentSpectrumType        m_HighEnergyIncidentSpectrum;
  IncidentSpectrumType        m_LowEnergyIncidentSpectrum;

private:
  DualEnergyNegativeLogLikelihood(const Self &); //purposely not implemented
  void operator = (const Self &); //purposely not implemented

};

}// namespace RTK

#endif
