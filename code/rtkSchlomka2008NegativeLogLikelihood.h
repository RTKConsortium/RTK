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

#ifndef rtkSchlomka2008NegativeLogLikelihood_h
#define rtkSchlomka2008NegativeLogLikelihood_h

#include "rtkProjectionsDecompositionNegativeLogLikelihood.h"

#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <itkVariableSizeMatrix.h>

namespace rtk
{
  /** \class rtkSchlomka2008NegativeLogLikelihood
   * \brief Cost function from the Schlomka 2008 PMB paper
   *
   * This class requires the method "Initialize()" to be run once, before it
   * is passed to the simplex minimizer
   * See the reference paper: "Experimental feasibility of multi-energy photon-counting
   * K-edge imaging in pre-clinical computed tomography", Schlomka et al, PMB 2008
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

// We have to define the cost function first
class Schlomka2008NegativeLogLikelihood : public rtk::ProjectionsDecompositionNegativeLogLikelihood
{
public:

  typedef Schlomka2008NegativeLogLikelihood                     Self;
  typedef rtk::ProjectionsDecompositionNegativeLogLikelihood    Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;
  itkNewMacro( Self );
  itkTypeMacro( Schlomka2008NegativeLogLikelihood, rtk::ProjectionsDecompositionNegativeLogLikelihood );

  typedef Superclass::ParametersType                ParametersType;
  typedef Superclass::DerivativeType                DerivativeType;
  typedef Superclass::MeasureType                   MeasureType;

  typedef Superclass::DetectorResponseType          DetectorResponseType;
  typedef Superclass::MaterialAttenuationsType      MaterialAttenuationsType;
  typedef Superclass::MeasuredDataType              MeasuredDataType;
  typedef Superclass::IncidentSpectrumType          IncidentSpectrumType;

  // Constructor
  Schlomka2008NegativeLogLikelihood()
  {
  m_NumberOfSpectralBins = 0;
  }

  // Destructor
  ~Schlomka2008NegativeLogLikelihood()
  {
  }

  void Initialize()
  {
  // This method computes the combined m_IncidentSpectrumAndDetectorResponseProduct
  // from m_DetectorResponse and m_IncidentSpectrum

  // In spectral CT, m_DetectorResponse has as many rows as the number of bins,
  // and m_IncidentSpectrum has only one row (there is only one spectrum illuminating
  // the object)
  m_IncidentSpectrumAndDetectorResponseProduct.set_size(m_DetectorResponse.rows(), m_DetectorResponse.cols());
  for (unsigned int i=0; i<m_DetectorResponse.rows(); i++)
    for (unsigned int j=0; j<m_DetectorResponse.cols(); j++)
      m_IncidentSpectrumAndDetectorResponseProduct[i][j] = m_DetectorResponse[i][j] * m_IncidentSpectrum[0][j];
  }

  // Not used with a simplex optimizer, but may be useful later
  // for gradient based methods
  void GetDerivative( const ParametersType & lineIntegrals,
                      DerivativeType & derivatives ) const ITK_OVERRIDE
  {
  // Set the size of the derivatives vector
  derivatives.set_size(m_NumberOfMaterials);

  // Get some required data
  vnl_vector<double> attenuationFactors;
  attenuationFactors.set_size(this->m_NumberOfEnergies);
  GetAttenuationFactors(lineIntegrals, attenuationFactors);
  vnl_vector<double> lambdas = ForwardModel(lineIntegrals);

  // Compute the vector of 1 - m_b / lambda_b
  vnl_vector<double> weights;
  weights.set_size(m_NumberOfSpectralBins);
  for (unsigned int i=0; i<m_NumberOfSpectralBins; i++)
    weights[i] = 1 - (m_MeasuredData[i] / lambdas[i]);

  // Prepare intermediate variables
  vnl_vector<double> intermediate_a;
  vnl_vector<double> partial_derivative_a;

  for (unsigned int a=0; a<m_NumberOfMaterials; a++)
    {
    // Compute the partial derivatives of lambda_b with respect to the material line integrals
    intermediate_a = element_product(-attenuationFactors, m_MaterialAttenuations.get_column(a));
    partial_derivative_a = m_IncidentSpectrumAndDetectorResponseProduct * intermediate_a;

    // Multiply them together element-wise, then dot product with the weights
    derivatives[a] = dot_product(partial_derivative_a,weights);
    }
  }

  // Main method
  MeasureType  GetValue( const ParametersType & parameters ) const
  {
  // Forward model: compute the expected number of counts in each bin
  vnl_vector<double> forward = ForwardModel(parameters);

  long double measure = 0;
  // Compute the negative log likelihood from the lambdas
  for (unsigned int i=0; i<m_NumberOfSpectralBins; i++)
    measure += forward[i] - std::log((long double)forward[i]) * m_MeasuredData[i];
  return measure;
  }

  void ComputeFischerMatrix(const ParametersType & lineIntegrals)
  {
  // Get some required data
  vnl_vector<double> attenuationFactors;
  attenuationFactors.set_size(this->m_NumberOfEnergies);
  GetAttenuationFactors(lineIntegrals, attenuationFactors);
  vnl_vector<double> lambdas = ForwardModel(lineIntegrals);

  // Compute the vector of m_b / lambda_b²
  vnl_vector<double> weights;
  weights.set_size(m_NumberOfSpectralBins);
  for (unsigned int i=0; i<m_NumberOfSpectralBins; i++)
    weights[i] = m_MeasuredData[i] / (lambdas[i] * lambdas[i]);

  // Prepare intermediate variables
  vnl_vector<double> intermediate_a;
  vnl_vector<double> intermediate_a_prime;
  vnl_vector<double> partial_derivative_a;
  vnl_vector<double> partial_derivative_a_prime;

  // Compute the Fischer information matrix
  m_Fischer.SetSize(m_NumberOfMaterials, m_NumberOfMaterials);
  for (unsigned int a=0; a<m_NumberOfMaterials; a++)
    {
    for (unsigned int a_prime=0; a_prime<m_NumberOfMaterials; a_prime++)
      {
      // Compute the partial derivatives of lambda_b with respect to the material line integrals
      intermediate_a = element_product(-attenuationFactors, m_MaterialAttenuations.get_column(a));
      intermediate_a_prime = element_product(-attenuationFactors, m_MaterialAttenuations.get_column(a_prime));

      partial_derivative_a = m_IncidentSpectrumAndDetectorResponseProduct * intermediate_a;
      partial_derivative_a_prime = m_IncidentSpectrumAndDetectorResponseProduct * intermediate_a_prime;

      // Multiply them together element-wise, then dot product with the weights
      partial_derivative_a_prime = element_product(partial_derivative_a, partial_derivative_a_prime);
      m_Fischer[a][a_prime] = dot_product(partial_derivative_a_prime,weights);
      }
    }
  }

private:
  Schlomka2008NegativeLogLikelihood(const Self &); //purposely not implemented
  void operator = (const Self &); //purposely not implemented

};

}// namespace RTK

#endif
