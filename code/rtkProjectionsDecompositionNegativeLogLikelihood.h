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
  itkTypeMacro( ProjectionsDecompositionNegativeLogLikelihood, SingleValuedCostFunction );

//  enum { SpaceDimension=m_NumberOfMaterials };

  typedef Superclass::ParametersType      ParametersType;
  typedef Superclass::DerivativeType      DerivativeType;
  typedef Superclass::MeasureType         MeasureType;

  typedef itk::VariableSizeMatrix<double>      DetectorResponseType;
  typedef itk::VariableSizeMatrix<double>      MaterialAttenuationsType;
  typedef itk::VariableLengthVector<double>    MeasuredDataType;
  typedef itk::VariableLengthVector<double>    IncidentSpectrumType;

  // Constructor
  ProjectionsDecompositionNegativeLogLikelihood()
  {
  m_NumberOfEnergies = 0;
  m_NumberOfMaterials = 0;
  }

  // Destructor
  ~ProjectionsDecompositionNegativeLogLikelihood()
  {
  }

  virtual vnl_vector<double> ForwardModel(const ParametersType & lineIntegrals) const = 0;
  virtual MeasureType  GetValue( const ParametersType & parameters ) const ITK_OVERRIDE = 0;
  virtual void GetDerivative( const ParametersType & lineIntegrals,
                      DerivativeType & derivatives ) const ITK_OVERRIDE = 0;

  unsigned int GetNumberOfParameters(void) const ITK_OVERRIDE
  {
  return m_NumberOfMaterials;
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

protected:
  MaterialAttenuationsType    m_MaterialAttenuations;
  DetectorResponseType        m_DetectorResponse;
  MeasuredDataType            m_MeasuredData;
  unsigned int                m_NumberOfEnergies;
  unsigned int                m_NumberOfMaterials;

private:
  ProjectionsDecompositionNegativeLogLikelihood(const Self &); //purposely not implemented
  void operator = (const Self &); //purposely not implemented

};

}// namespace RTK

#endif
