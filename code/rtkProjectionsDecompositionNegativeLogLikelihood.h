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

#ifndef rtkProjectionsDecompositionNegativeLogLikelihood_h
#define rtkProjectionsDecompositionNegativeLogLikelihood_h

#include <itkSingleValuedCostFunction.h>
#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <itkVariableSizeMatrix.h>

namespace rtk
{
  /** \class rtkProjectionsDecompositionNegativeLogLikelihood
   * \brief Base class for projections decomposition cost functions
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

// We have to define the cost function first
class ProjectionsDecompositionNegativeLogLikelihood : public itk::SingleValuedCostFunction
{
public:

  typedef ProjectionsDecompositionNegativeLogLikelihood     Self;
  typedef itk::SingleValuedCostFunction                     Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  itkNewMacro( Self );
  itkTypeMacro( ProjectionsDecompositionNegativeLogLikelihood, SingleValuedCostFunction );

//  enum { SpaceDimension=m_NumberOfMaterials };

  typedef Superclass::ParametersType      ParametersType;
  typedef Superclass::DerivativeType      DerivativeType;
  typedef Superclass::MeasureType         MeasureType;

  typedef vnl_matrix<double>                        DetectorResponseType;
  typedef vnl_matrix<double>                        MaterialAttenuationsType;
  typedef vnl_matrix<float>                         IncidentSpectrumType;
  typedef itk::VariableLengthVector<double>         MeasuredDataType;
  typedef itk::VariableLengthVector<unsigned int>   ThresholdsType;
  typedef itk::VariableSizeMatrix<double>           MeanAttenuationInBinType;

  // Constructor
  ProjectionsDecompositionNegativeLogLikelihood()
  {
  m_NumberOfEnergies = 0;
  m_NumberOfMaterials = 0;
  m_Initialized = false;
  }

  // Destructor
  ~ProjectionsDecompositionNegativeLogLikelihood()
  {
  }

  virtual MeasureType GetValue( const ParametersType & parameters ) const ITK_OVERRIDE {
  long double measure = 0;
  return measure;
  }
  virtual void GetDerivative( const ParametersType & lineIntegrals,
                      DerivativeType & derivatives ) const ITK_OVERRIDE {}
  virtual void Initialize() {}

  virtual itk::VariableLengthVector<float> GetInverseCramerRaoLowerBound()
  {
  // Return the inverses of the diagonal components (i.e. the inverse variances, to be used directly in WLS reconstruction)
  itk::VariableLengthVector<double> diag;
  diag.SetSize(m_NumberOfMaterials);
  diag.Fill(0);

  for (unsigned int mat=0; mat<m_NumberOfMaterials; mat++)
    diag[mat] = 1./m_Fischer.GetInverse()[mat][mat];
  return diag;
  }

  virtual itk::VariableLengthVector<float> GetFischerMatrix()
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

  virtual void ComputeFischerMatrix(const ParametersType & lineIntegrals){}

  unsigned int GetNumberOfParameters(void) const ITK_OVERRIDE
  {
  return m_NumberOfMaterials;
  }

  virtual vnl_vector<double> ForwardModel(const ParametersType & lineIntegrals) const
  {
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
      for (unsigned int energy=m_Thresholds[bin]-1; (energy<m_Thresholds[bin+1]) && (energy < this->m_MaterialAttenuations.rows()); energy++)
        {
        accumulate += this->m_MaterialAttenuations[energy][mat] * this->m_IncidentSpectrum[0][energy];
        accumulateWeights += this->m_IncidentSpectrum[0][energy];
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

  virtual vnl_vector<double>  GetVariances( const ParametersType & lineIntegrals ) const
  {
  vnl_vector<double> meaninglessResult;
  meaninglessResult.set_size(m_NumberOfSpectralBins);
  meaninglessResult.fill(0.);
  return(meaninglessResult);
  }

  itkSetMacro(MeasuredData, MeasuredDataType)
  itkGetMacro(MeasuredData, MeasuredDataType)

  itkSetMacro(DetectorResponse, DetectorResponseType)
  itkGetMacro(DetectorResponse, DetectorResponseType)

  itkSetMacro(MaterialAttenuations, MaterialAttenuationsType)
  itkGetMacro(MaterialAttenuations, MaterialAttenuationsType)

  itkSetMacro(NumberOfEnergies, unsigned int)
  itkGetMacro(NumberOfEnergies, unsigned int)

  itkSetMacro(NumberOfMaterials, unsigned int)
  itkGetMacro(NumberOfMaterials, unsigned int)

  itkSetMacro(IncidentSpectrum, IncidentSpectrumType)
  itkGetMacro(IncidentSpectrum, IncidentSpectrumType)

  itkSetMacro(NumberOfSpectralBins, unsigned int)
  itkGetMacro(NumberOfSpectralBins, unsigned int)

  itkSetMacro(Thresholds, ThresholdsType)
  itkGetMacro(Thresholds, ThresholdsType)

protected:
  MaterialAttenuationsType          m_MaterialAttenuations;
  DetectorResponseType              m_DetectorResponse;
  MeasuredDataType                  m_MeasuredData;
  ThresholdsType                    m_Thresholds;
  IncidentSpectrumType              m_IncidentSpectrum;
  vnl_matrix<double>                m_IncidentSpectrumAndDetectorResponseProduct;
  unsigned int                      m_NumberOfEnergies;
  unsigned int                      m_NumberOfMaterials;
  unsigned int                      m_NumberOfSpectralBins;
  bool                              m_Initialized;
  itk::VariableSizeMatrix<float>    m_Fischer;

private:
  ProjectionsDecompositionNegativeLogLikelihood(const Self &); //purposely not implemented
  void operator = (const Self &); //purposely not implemented

};

}// namespace RTK

#endif
