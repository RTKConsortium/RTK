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
  typedef itk::VariableLengthVector<unsigned int>   ThresholdsType;
  typedef itk::VariableSizeMatrix<double>           MeanAttenuationInBinType;


  // Constructor
  Schlomka2008NegativeLogLikelihood()
  {
  m_NumberOfSpectralBins = 0;
  m_IsSpectralCT = false;
  m_IsDualEnergyCT = false;
  }

  // Destructor
  ~Schlomka2008NegativeLogLikelihood()
  {
  }

  itk::VariableLengthVector<double> GuessInitialization() const
  {
  itk::VariableLengthVector<double> initialGuess;
  initialGuess.SetSize(m_NumberOfMaterials);

  // Compute the mean attenuation in each bin, weighted by the input spectrum
  // Needs to be done for each pixel, since the input spectrum is variable
  MeanAttenuationInBinType MeanAttenuationInBin;
  MeanAttenuationInBin.SetSize(this->m_NumberOfMaterials, this->m_NumberOfSpectralBins);
  MeanAttenuationInBin.Fill(0);

  for (unsigned int mat = 0; mat<this->m_NumberOfMaterials; mat++)
    {
    for (unsigned int bin=0; bin<m_NumberOfSpectralBins; bin++)
      {
      double accumulate = 0;
      double accumulateWeights = 0;
      if (m_IsSpectralCT)
        {
        for (unsigned int energy=m_Thresholds[bin]-1; (energy<m_Thresholds[bin+1]) && (energy < this->m_MaterialAttenuations.rows()); energy++)
          {
          accumulate += this->m_MaterialAttenuations[energy][mat] * this->m_IncidentSpectrum[0][energy];
          accumulateWeights += this->m_IncidentSpectrum[0][energy];
          }
        }
      if (m_IsDualEnergyCT) // In this case there are no real bins, but we still use almost the same method
        {
        for (unsigned int energy=0; energy<m_NumberOfEnergies; energy++)
          {
          accumulate += this->m_MaterialAttenuations[energy][mat] * this->m_IncidentSpectrum[bin][energy];
          accumulateWeights += this->m_IncidentSpectrum[bin][energy];
          }
        }

      MeanAttenuationInBin[mat][bin] = accumulate / accumulateWeights;
      }
    }

  for (unsigned int mat = 0; mat<m_NumberOfMaterials; mat++)
    {
    // Initialise to a very high value
    initialGuess[mat] = 1e10;
    for (unsigned int bin = 0; bin<m_NumberOfSpectralBins; bin++)
      {
      // Compute the length of current material required to obtain the attenuation
      // observed in current bin. Keep only the minimum among all bins
      double requiredLength = this->BinwiseLogTransform()[bin] / MeanAttenuationInBin[mat][bin];
      if (initialGuess[mat] > requiredLength)
        initialGuess[mat] = requiredLength;
      }
    }

  return initialGuess;
  }

  itk::VariableLengthVector<double> BinwiseLogTransform() const
  {
  itk::VariableLengthVector<double> logTransforms;
  logTransforms.SetSize(m_NumberOfSpectralBins);

  vnl_vector<double> ones, nonAttenuated;
  ones.set_size(m_NumberOfEnergies);
  ones.fill(1.0);

  // The way m_IncidentSpectrumAndDetectorResponseProduct works is
  // it is mutliplied by the vector of attenuations factors (here
  // filled with ones, since we want the non-attenuated signal)
  nonAttenuated = m_IncidentSpectrumAndDetectorResponseProduct * ones;

  for (unsigned int i=0; i<m_MeasuredData.GetSize(); i++)
    {
    // Divide by the actually measured photon counts and apply log
    if (m_MeasuredData[i] > 0)
      logTransforms[i] = log(nonAttenuated[i] / m_MeasuredData[i]);
    }

  return logTransforms;
  }

  void Initialize()
  {
  // This method computes the combined m_IncidentSpectrumAndDetectorResponseProduct
  // from m_DetectorResponse and m_IncidentSpectrum

  // First, determine whether we are in the spectral of dual energy CT case
  if( (m_DetectorResponse.rows() == m_NumberOfSpectralBins) && (m_IncidentSpectrum.rows() == 1) && (m_DetectorResponse.cols() == m_IncidentSpectrum.cols()) )
    {
    m_IsSpectralCT = true;
    m_IsDualEnergyCT = false;
    }
  if( (m_DetectorResponse.rows() == 1) && (m_IncidentSpectrum.rows() == 2) && (m_DetectorResponse.cols() == m_IncidentSpectrum.cols()) )
    {
    m_IsSpectralCT = false;
    m_IsDualEnergyCT = true;
    }

  // In spectral CT, m_DetectorResponse has as many rows as the number of bins,
  // and m_IncidentSpectrum has only one row (there is only one spectrum illuminating
  // the object)
  if(m_IsSpectralCT) // Spectral CT
    {
    m_IncidentSpectrumAndDetectorResponseProduct.set_size(m_DetectorResponse.rows(), m_DetectorResponse.cols());
    for (unsigned int i=0; i<m_DetectorResponse.rows(); i++)
      for (unsigned int j=0; j<m_DetectorResponse.cols(); j++)
        m_IncidentSpectrumAndDetectorResponseProduct[i][j] = m_DetectorResponse[i][j] * m_IncidentSpectrum[0][j];
    }

  // In dual energy CT, one possible design is to illuminate the object with
  // either a low energy or a high energy spectrum, alternating between the two. In that case
  // m_DetectorResponse has only one row (there is a single detector) and m_IncidentSpectrum
  // has two rows (one for high energy, the other for low)
  if(m_IsDualEnergyCT) // Dual energy CT
    {
    // Even though in this case, there are no bins, set the number of bins to two
    // so as to be able to use methods designed for spectral CT later on
    m_NumberOfSpectralBins = 2;

    m_IncidentSpectrumAndDetectorResponseProduct.set_size(2, m_DetectorResponse.cols());
    for (unsigned int i=0; i<2; i++)
      for (unsigned int j=0; j<m_DetectorResponse.cols(); j++)
        m_IncidentSpectrumAndDetectorResponseProduct[i][j] = m_DetectorResponse[0][j] * m_IncidentSpectrum[i][j];
    }

  // An alternative dual energy CT design is the so-called sandwich detector, but it is
  // already covered by the spectral CT case
  }

  vnl_vector<double> ForwardModel(const ParametersType & lineIntegrals) const ITK_OVERRIDE
  {
  // Variable length vector and variable size matrix cannot be used in linear algebra operations
  // Get their vnl counterparts, which can
  vnl_vector<double> attenuationFactors;
  attenuationFactors.set_size(m_NumberOfEnergies);
  GetAttenuationFactors(lineIntegrals, attenuationFactors);

  // Apply detector response, getting the lambdas
  return (m_IncidentSpectrumAndDetectorResponseProduct * attenuationFactors);
  }

  void GetAttenuationFactors(const ParametersType & lineIntegrals, vnl_vector<double> & attenuationFactors) const
  {
  // Apply attenuation at each energy
  vnl_vector<double> vnlLineIntegrals;

  // Initialize the line integrals vnl vector
  vnlLineIntegrals.set_size(m_NumberOfMaterials);
  for (unsigned int m=0; m<m_NumberOfMaterials; m++)
    vnlLineIntegrals[m] = lineIntegrals[m];

  // Apply the material attenuations matrix
  attenuationFactors = this->m_MaterialAttenuations * vnlLineIntegrals;

  // Compute the negative exponential
  for (unsigned int energy = 0; energy<m_NumberOfEnergies; energy++)
    {
    attenuationFactors[energy] = std::exp(-attenuationFactors[energy]);
    }
  }

  itk::VariableLengthVector<float> GetInverseCramerRaoLowerBound()
  {
  // Return the inverses of the diagonal components (i.e. the inverse variances, to be used directly in WLS reconstruction)
  itk::VariableLengthVector<double> diag;
  diag.SetSize(m_NumberOfMaterials);
  diag.Fill(0);

  for (unsigned int mat=0; mat<m_NumberOfMaterials; mat++)
    diag[mat] = 1./m_Fischer.GetInverse()[mat][mat];
  return diag;
  }

  itk::VariableLengthVector<float> GetFischerMatrix()
  {
  // Return the whole Fischer information matrix
  itk::VariableLengthVector<double> fischer;
  fischer.SetSize(m_NumberOfMaterials * m_NumberOfMaterials);
  fischer.Fill(0);

  for (unsigned int i=0; i<m_NumberOfMaterials; i++)
    for (unsigned int j=0; j<m_NumberOfMaterials; j++)
    fischer[i * m_NumberOfMaterials + j] = m_Fischer[i][j];
  return fischer;
  }

  // Not used with a simplex optimizer, but may be useful later
  // for gradient based methods
  void GetDerivative( const ParametersType & lineIntegrals,
                      DerivativeType & derivatives ) const ITK_OVERRIDE
  {
//  // Set the size of the derivatives vector
//  derivatives.set_size(m_NumberOfMaterials);

//  // Get some required data
//  vnl_vector<double> attenuatedIncidentSpectrum(GetAttenuatedIncidentSpectrum(lineIntegrals).GetDataPointer(), GetAttenuatedIncidentSpectrum(lineIntegrals).GetSize());
//  vnl_vector<double> lambdas = ForwardModel(lineIntegrals);

//  // Compute the vector of 1 - m_b / lambda_b
//  vnl_vector<double> weights;
//  weights.set_size(m_NumberOfSpectralBins);
//  for (unsigned int i=0; i<m_NumberOfSpectralBins; i++)
//    weights[i] = 1 - (m_MeasuredData[i] / lambdas[i]);

//  // Prepare intermediate variables
//  vnl_vector<double> intermediate_a;
//  vnl_vector<double> partial_derivative_a;

//  for (unsigned int a=0; a<m_NumberOfMaterials; a++)
//    {
//    // Compute the partial derivatives of lambda_b with respect to the material line integrals
//    intermediate_a = element_product(-attenuatedIncidentSpectrum, m_MaterialAttenuations.GetVnlMatrix().get_row(a));
//    partial_derivative_a = m_DetectorResponse.GetVnlMatrix() * intermediate_a;

//    // Multiply them together element-wise, then dot product with the weights
//    derivatives[a] = dot_product(partial_derivative_a,weights);
//    }
  }

  // Main method
  MeasureType  GetValue( const ParametersType & parameters ) const ITK_OVERRIDE
  {
  // Forward model: compute the expected number of counts in each bin
  vnl_vector<double> forward = ForwardModel(parameters);

  long double measure = 0;
  if (m_IsSpectralCT)
    {
  // Compute the negative log likelihood from the lambdas
  for (unsigned int i=0; i<m_NumberOfSpectralBins; i++)
    measure += forward[i] - std::log((long double)forward[i]) * m_MeasuredData[i];
    }
  if (m_IsDualEnergyCT)
    {
    // TODO: Improve this estimation
    // We assume that the variance of the integrated energy is equal to the mean
    // From equation (5) of "Cramér–Rao lower bound of basis image noise in multiple-energy x-ray imaging", PMB 2009, Roessl et al,
    // we replace the variance with the mean

    // Compute the negative log likelihood from the expectedEnergies
    for (unsigned int i=0; i<this->m_NumberOfMaterials; i++)
      measure += std::log((long double)forward[i]) + (forward[i] - this->m_MeasuredData[i]) * (forward[i] - this->m_MeasuredData[i]) / forward[i];
    measure *= 0.5;
    }

  return measure;
  }

//  void ComputeFischerMatrix(const ParametersType & lineIntegrals)
//  {
//  // Get some required data
//  vnl_vector<double> attenuatedIncidentSpectrum(GetAttenuatedIncidentSpectrum(lineIntegrals).GetDataPointer(), GetAttenuatedIncidentSpectrum(lineIntegrals).GetSize());
//  vnl_vector<double> lambdas = ForwardModel(lineIntegrals);

//  // Compute the vector of m_b / lambda_b²
//  vnl_vector<double> weights;
//  weights.set_size(m_NumberOfSpectralBins);
//  for (unsigned int i=0; i<m_NumberOfSpectralBins; i++)
//    weights[i] = m_MeasuredData[i] / (lambdas[i] * lambdas[i]);

//  // Prepare intermediate variables
//  vnl_vector<double> intermediate_a;
//  vnl_vector<double> intermediate_a_prime;
//  vnl_vector<double> partial_derivative_a;
//  vnl_vector<double> partial_derivative_a_prime;

//  // Compute the Fischer information matrix
//  m_Fischer.SetSize(m_NumberOfMaterials, m_NumberOfMaterials);
//  for (unsigned int a=0; a<m_NumberOfMaterials; a++)
//    {
//    for (unsigned int a_prime=0; a_prime<m_NumberOfMaterials; a_prime++)
//      {
//      // Compute the partial derivatives of lambda_b with respect to the material line integrals
//      intermediate_a = element_product(attenuatedIncidentSpectrum, m_MaterialAttenuations.GetVnlMatrix().get_row(a));
//      intermediate_a_prime = element_product(attenuatedIncidentSpectrum, m_MaterialAttenuations.GetVnlMatrix().get_row(a_prime));

//      partial_derivative_a = m_DetectorResponse.GetVnlMatrix() * intermediate_a;
//      partial_derivative_a_prime = m_DetectorResponse.GetVnlMatrix() * intermediate_a_prime;

//      // Multiply them together element-wise, then dot product with the weights
//      partial_derivative_a_prime = element_product(partial_derivative_a, partial_derivative_a_prime);
//      m_Fischer[a][a_prime] = dot_product(partial_derivative_a_prime,weights);
//      }
//    }
//  }


  itkSetMacro(IncidentSpectrum, IncidentSpectrumType)
  itkGetMacro(IncidentSpectrum, IncidentSpectrumType)

  itkSetMacro(NumberOfSpectralBins, unsigned int)
  itkGetMacro(NumberOfSpectralBins, unsigned int)

  itkSetMacro(Thresholds, ThresholdsType)
  itkGetMacro(Thresholds, ThresholdsType)

protected:
  IncidentSpectrumType              m_IncidentSpectrum;
  vnl_matrix<double>                m_IncidentSpectrumAndDetectorResponseProduct;
  unsigned int                      m_NumberOfSpectralBins;
  itk::VariableSizeMatrix<float>    m_Fischer;
  ThresholdsType                    m_Thresholds;
  bool                              m_IsSpectralCT, m_IsDualEnergyCT;

private:
  Schlomka2008NegativeLogLikelihood(const Self &); //purposely not implemented
  void operator = (const Self &); //purposely not implemented

};

}// namespace RTK

#endif
