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

#ifndef rtkAlvarez1976NegativeLogLikelihood_h
#define rtkAlvarez1976NegativeLogLikelihood_h

#include "rtkProjectionsDecompositionNegativeLogLikelihood.h"

#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <itkVariableSizeMatrix.h>

namespace rtk
{
  /** \class rtkAlvarez1976NegativeLogLikelihood
   * \brief Cost function from the Alvarez 1976 PMB paper
   *
   * See the reference paper: "Energy-selective reconstructions in
   * X-Ray Computerized Tomography", Alvarez and Macovski, PMB 1976
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

// We have to define the cost function first
class Alvarez1976NegativeLogLikelihood : public rtk::ProjectionsDecompositionNegativeLogLikelihood
{
public:

  typedef Alvarez1976NegativeLogLikelihood                      Self;
  typedef rtk::ProjectionsDecompositionNegativeLogLikelihood    Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;
  itkNewMacro( Self );
  itkTypeMacro( Alvarez1976NegativeLogLikelihood, rtk::ProjectionsDecompositionNegativeLogLikelihood );

  typedef Superclass::ParametersType          ParametersType;
  typedef Superclass::DerivativeType          DerivativeType;
  typedef Superclass::MeasureType             MeasureType;

  typedef Superclass::DetectorResponseType      DetectorResponseType;
  typedef Superclass::MaterialAttenuationsType  MaterialAttenuationsType;
  typedef Superclass::MeasuredDataType          MeasuredDataType;
  typedef Superclass::IncidentSpectrumType      IncidentSpectrumType;

  // Constructor
  Alvarez1976NegativeLogLikelihood()
  {
  }

  // Destructor
  ~Alvarez1976NegativeLogLikelihood()
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

  // Apply detector response, getting the expectedEnergies
  vnl_vector<double> expectedEnergies;
  expectedEnergies.set_size(this->m_NumberOfMaterials);
  expectedEnergies[0] = (m_DetectorResponse.GetVnlMatrix() * vnl_vec_he).sum(); // There should only be one term, the sum is just for type correctness
  expectedEnergies[1] = (m_DetectorResponse.GetVnlMatrix() * vnl_vec_le).sum();
  return expectedEnergies;
  }

  itk::VariableLengthVector<double> GetAttenuatedIncidentSpectrum(IncidentSpectrumType incident, const ParametersType & lineIntegrals) const
  {
  // Solid angle of detector pixel, exposure time and mAs should already be
  // taken into account in the incident spectrum image

  if(m_NumberOfEnergies != incident.GetSize())
      itkGenericExceptionMacro(<< "Incident spectrum does not have the correct size")

  // Apply attenuation at each energy
  itk::VariableLengthVector<double> attenuatedIncidentSpectrum;
  attenuatedIncidentSpectrum.SetSize(m_NumberOfEnergies);
  attenuatedIncidentSpectrum.Fill(0);
  for (unsigned int e=0; e<m_NumberOfEnergies; e++)
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
  // Forward model: compute the expectedEnergies integrated by the detector
  vnl_vector<double> expectedEnergies = ForwardModel(parameters);

  // Compute the negative log likelihood from the expectedEnergies
  long double measure = 0;
  for (unsigned int i=0; i<this->m_NumberOfMaterials; i++)
    measure += expectedEnergies[i] - std::log((long double)expectedEnergies[i]) * this->m_MeasuredData[i];
  return measure;
  }

  itk::VariableLengthVector<double> GetInverseCramerRaoLowerBound(const ParametersType & lineIntegrals) const
  {
  // Dummy function at the moment, returns a vector filled with zeros
  itk::VariableLengthVector<double> diag;
  diag.SetSize(m_NumberOfMaterials);
  diag.Fill(0);
  return diag;
  }

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
  Alvarez1976NegativeLogLikelihood(const Self &); //purposely not implemented
  void operator = (const Self &); //purposely not implemented

};

}// namespace RTK

#endif
