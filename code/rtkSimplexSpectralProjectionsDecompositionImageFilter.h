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

#include <itkImageToImageFilter.h>
#include <itkAmoebaOptimizer.h>
#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <itkVariableSizeMatrix.h>

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

// We have to define the cost function first
class Schlomka2008NegativeLogLikelihood : public itk::SingleValuedCostFunction
{
public:

  typedef Schlomka2008NegativeLogLikelihood   Self;
  typedef itk::SingleValuedCostFunction       Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;
  itkNewMacro( Self );
  itkTypeMacro( Schlomka2008NegativeLogLikelihood, SingleValuedCostFunction );

//  enum { SpaceDimension=m_NumberOfMaterials };

  typedef Superclass::ParametersType      ParametersType;
  typedef Superclass::DerivativeType      DerivativeType;
  typedef Superclass::MeasureType         MeasureType;

  typedef itk::VariableSizeMatrix<float>      DetectorResponseType;
  typedef itk::VariableSizeMatrix<float>      MaterialAttenuationsType;
  typedef itk::VariableLengthVector<float>    DetectorCountsType;
  typedef itk::VariableLengthVector<float>    IncidentSpectrumType;

  // Constructor
  Schlomka2008NegativeLogLikelihood()
  {
  }

  // Destructor
  ~Schlomka2008NegativeLogLikelihood()
  {
  }

  vnl_vector<float> ForwardModel(const ParametersType & lineIntegrals) const
  {
  // Variable length vector and variable size matrix cannot be used in linear algebra operations
  // Get their vnl counterparts, which can
  vnl_vector<float> vnl_vec(GetAttenuatedIncidentSpectrum(lineIntegrals).GetDataPointer(), GetAttenuatedIncidentSpectrum(lineIntegrals).GetSize());

  // Apply detector response, getting the lambdas
  return (m_DetectorResponse.GetVnlMatrix() * vnl_vec);
  }

  itk::VariableLengthVector<float> GetAttenuatedIncidentSpectrum(const ParametersType & lineIntegrals) const
  {
  // Solid angle of detector pixel, exposure time and mAs should already be
  // taken into account in the incident spectrum image

  // Apply attenuation at each energy
  itk::VariableLengthVector<float> attenuatedIncidentSpectrum;
  attenuatedIncidentSpectrum.SetSize(m_NumberOfEnergies);
  attenuatedIncidentSpectrum.Fill(0);
  for (unsigned int e=0; e<m_NumberOfEnergies; e++)
    {
    float totalAttenuation = 0.;
    for (unsigned int m=0; m<m_NumberOfMaterials; m++)
      {
      totalAttenuation += lineIntegrals[m] * m_MaterialAttenuations[m][e];
      }

    attenuatedIncidentSpectrum[e] = m_IncidentSpectrum[e] * std::exp(-totalAttenuation);
    }

  return attenuatedIncidentSpectrum;
  }

  itk::VariableLengthVector<float> GetInverseCramerRaoLowerBound(const ParametersType & lineIntegrals) const
  {
  // Get some required data
  vnl_vector<float> attenuatedIncidentSpectrum(GetAttenuatedIncidentSpectrum(lineIntegrals).GetDataPointer(), GetAttenuatedIncidentSpectrum(lineIntegrals).GetSize());
  vnl_vector<float> lambdas = ForwardModel(lineIntegrals);

  // Compute the vector of m_b / lambda_b²
  vnl_vector<float> weights;
  weights.set_size(m_NumberOfSpectralBins);
  for (unsigned int i=0; i<m_NumberOfSpectralBins; i++)
    weights[i] = m_DetectorCounts[i] / (lambdas[i] * lambdas[i]);

  // Prepare intermediate variables
  vnl_vector<float> intermediate_a;
  vnl_vector<float> intermediate_a_prime;
  vnl_vector<float> partial_derivative_a;
  vnl_vector<float> partial_derivative_a_prime;

  // Compute the Fischer information matrix
  itk::VariableSizeMatrix<float> Fischer;
  Fischer.SetSize(m_NumberOfMaterials, m_NumberOfMaterials);
  for (unsigned int a=0; a<m_NumberOfMaterials; a++)
    {
    for (unsigned int a_prime=0; a_prime<m_NumberOfMaterials; a_prime++)
      {
      // Compute the partial derivatives of lambda_b with respect to the material line integrals
      intermediate_a = element_product(attenuatedIncidentSpectrum, m_MaterialAttenuations.GetVnlMatrix().get_row(a));
      intermediate_a_prime = element_product(attenuatedIncidentSpectrum, m_MaterialAttenuations.GetVnlMatrix().get_row(a_prime));

      partial_derivative_a = m_DetectorResponse.GetVnlMatrix() * intermediate_a;
      partial_derivative_a_prime = m_DetectorResponse.GetVnlMatrix() * intermediate_a_prime;

      // Multiply them together element-wise, then dot product with the weights
      partial_derivative_a_prime = element_product(partial_derivative_a, partial_derivative_a_prime);
      Fischer[a][a_prime] = dot_product(partial_derivative_a_prime,weights);
      }
    }

  // Invert the Fischer matrix
  itk::VariableLengthVector<float> diag;
  diag.SetSize(m_NumberOfMaterials);
  diag.Fill(0);

  Fischer = Fischer.GetInverse();

  // Return the inverses of the diagonal components (i.e. the inverse variances, to be used directly in WLS reconstruction)
  for (unsigned int mat=0; mat<m_NumberOfMaterials; mat++)
    diag[mat] = 1./Fischer[mat][mat];
  return diag;
  }

  // Not implemented, since it is too complex to compute
  // Therefore we will only use a zero-th order method
  void GetDerivative( const ParametersType & ,
                      DerivativeType &  ) const
  {
  }

  // Main method
  MeasureType  GetValue( const ParametersType & parameters ) const
  {
  // Forward model: compute the expected number of counts in each bin
  vnl_vector<float> lambdas = ForwardModel(parameters);

  // Compute the negative log likelihood from the lambdas
  MeasureType measure = 0;
  for (unsigned int i=0; i<m_NumberOfSpectralBins; i++)
    measure += lambdas[i] - std::log(lambdas[i]) * m_DetectorCounts[i];

  return measure;
  }

  unsigned int GetNumberOfParameters(void) const
  {
  return m_NumberOfMaterials;
  }

  itkSetMacro(DetectorCounts, DetectorCountsType)
  itkGetMacro(DetectorCounts, DetectorCountsType)

  itkSetMacro(DetectorResponse, DetectorResponseType)
  itkGetMacro(DetectorResponse, DetectorResponseType)

  itkSetMacro(IncidentSpectrum, IncidentSpectrumType)
  itkGetMacro(IncidentSpectrum, IncidentSpectrumType)

  itkSetMacro(MaterialAttenuations, MaterialAttenuationsType)
  itkGetMacro(MaterialAttenuations, MaterialAttenuationsType)

  itkSetMacro(NumberOfEnergies, unsigned int)
  itkGetMacro(NumberOfEnergies, unsigned int)

  itkSetMacro(NumberOfSpectralBins, unsigned int)
  itkGetMacro(NumberOfSpectralBins, unsigned int)

  itkSetMacro(NumberOfMaterials, unsigned int)
  itkGetMacro(NumberOfMaterials, unsigned int)

protected:
  MaterialAttenuationsType    m_MaterialAttenuations;
  DetectorResponseType        m_DetectorResponse;
  IncidentSpectrumType        m_IncidentSpectrum;
  DetectorCountsType          m_DetectorCounts;
  unsigned int                m_NumberOfEnergies;
  unsigned int                m_NumberOfSpectralBins;
  unsigned int                m_NumberOfMaterials;

private:
  Schlomka2008NegativeLogLikelihood(const Self &); //purposely not implemented
  void operator = (const Self &); //purposely not implemented

};


template<typename DecomposedProjectionsType,
         typename SpectralProjectionsType,
         typename IncidentSpectrumImageType = itk::VectorImage<float, 2>,
         typename DetectorResponseImageType = itk::Image<float, 2>,
         typename MaterialAttenuationsImageType = itk::Image<float, 2> >
class ITK_EXPORT SimplexSpectralProjectionsDecompositionImageFilter :
  public itk::ImageToImageFilter<DecomposedProjectionsType, DecomposedProjectionsType>
{
public:
  /** Standard class typedefs. */
  typedef SimplexSpectralProjectionsDecompositionImageFilter                                Self;
  typedef itk::ImageToImageFilter<DecomposedProjectionsType, DecomposedProjectionsType>     Superclass;
  typedef itk::SmartPointer<Self>                                                           Pointer;
  typedef itk::SmartPointer<const Self>                                                     ConstPointer;

  /** Some convenient typedefs. */
  typedef SpectralProjectionsType       InputImageType;
  typedef SpectralProjectionsType       OutputImageType;

  /** Convenient information */
  typedef itk::VariableSizeMatrix<float>            DetectorResponseType;
  typedef itk::VariableSizeMatrix<float>            MaterialAttenuationsType;
  typedef itk::VariableLengthVector<unsigned int>   ThresholdsType;

  /** Typedefs of each subfilter of this composite filter */
  typedef Schlomka2008NegativeLogLikelihood                             CostFunctionType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(SimplexSpectralProjectionsDecompositionImageFilter, itk::ImageToImageFilter)

  /** Get / Set the number of iterations. Default is 300. */
  itkGetMacro(NumberOfIterations, unsigned int)
  itkSetMacro(NumberOfIterations, unsigned int)

  /** Set/Get the input material-decomposed stack of projections (only used for initialization) */
  void SetInputDecomposedProjections(const DecomposedProjectionsType* DecomposedProjections);
  typename DecomposedProjectionsType::ConstPointer GetInputDecomposedProjections();

  /** Set/Get the input stack of spectral projections (to be decomposed in materials) */
  void SetInputSpectralProjections(const SpectralProjectionsType* SpectralProjections);
  typename SpectralProjectionsType::ConstPointer GetInputSpectralProjections();

  /** Set/Get the incident spectrum input image */
  void SetInputIncidentSpectrum(const IncidentSpectrumImageType* IncidentSpectrum);
  typename IncidentSpectrumImageType::ConstPointer GetInputIncidentSpectrum();

  /** Set/Get the detector response as an image */
  void SetDetectorResponse(const DetectorResponseImageType* DetectorResponse);
  typename DetectorResponseImageType::ConstPointer GetDetectorResponse();

  /** Set/Get the material attenuations as an image */
  void SetMaterialAttenuations(const MaterialAttenuationsImageType* MaterialAttenuations);
  typename MaterialAttenuationsImageType::ConstPointer GetMaterialAttenuations();

  itkSetMacro(Thresholds, ThresholdsType)
  itkGetMacro(Thresholds, ThresholdsType)

  itkSetMacro(NumberOfEnergies, unsigned int)
  itkGetMacro(NumberOfEnergies, unsigned int)

  itkSetMacro(NumberOfSpectralBins, unsigned int)
  itkGetMacro(NumberOfSpectralBins, unsigned int)

  itkSetMacro(NumberOfMaterials, unsigned int)
  itkGetMacro(NumberOfMaterials, unsigned int)

protected:
  SimplexSpectralProjectionsDecompositionImageFilter();
  ~SimplexSpectralProjectionsDecompositionImageFilter() ITK_OVERRIDE {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;
  void ThreadedGenerateData(const typename DecomposedProjectionsType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;

  /**  Create the Output */
  typedef itk::ProcessObject::DataObjectPointerArraySizeType DataObjectPointerArraySizeType;
  using Superclass::MakeOutput;
  itk::DataObject::Pointer MakeOutput(DataObjectPointerArraySizeType idx) ITK_OVERRIDE;

  /** The inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  MaterialAttenuationsType   m_MaterialAttenuations;
  DetectorResponseType       m_DetectorResponse;
  ThresholdsType             m_Thresholds;
  unsigned int               m_NumberOfEnergies;
  unsigned int               m_NumberOfSpectralBins;
  unsigned int               m_NumberOfMaterials;

  /** Number of simplex iterations */
  unsigned int m_NumberOfIterations;

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
