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

#ifndef rtkSimplexProjectionsDecompositionImageFilter_hxx
#define rtkSimplexProjectionsDecompositionImageFilter_hxx

#include "rtkSimplexProjectionsDecompositionImageFilter.h"
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

namespace rtk
{

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SimplexProjectionsDecompositionImageFilter()
{
  this->SetNumberOfIndexedOutputs(2); // decomposed projections, inverse variance of decomposition noise

  // Decomposed projections
  this->SetNthOutput( 0, this->MakeOutput( 0 ) );

  // Inverse variance of decomposition noise
  this->SetNthOutput( 1, this->MakeOutput( 1 ) );

  // Fischer matrix (estimate of inverse covariance of decomposition)
  this->SetNthOutput( 2, this->MakeOutput( 2 ) );

  // Set the default values of member parameters
  m_NumberOfIterations=300;
  m_NumberOfMaterials = 4;
  m_NumberOfEnergies = 100;
  m_OptimizeWithRestarts = false;
  m_RescaleAttenuations = false;

  // Fill in the vectors and matrices with zeros
  m_MaterialAttenuations.Fill(0.); //Not sure this works
  m_RescaledMaterialAttenuations.Fill(0.);
  m_DetectorResponse.Fill(0.);
  m_RescalingFactors.Fill(0.);
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputDecomposedProjections(const DecomposedProjectionsType* DecomposedProjections)
{
  this->SetNthInput(0, const_cast<DecomposedProjectionsType*>(DecomposedProjections));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputMeasuredProjections(const MeasuredProjectionsType *SpectralProjections)
{
  this->SetInput("MeasuredProjections", const_cast<MeasuredProjectionsType*>(SpectralProjections));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetDetectorResponse(const DetectorResponseImageType *DetectorResponse)
{
  this->SetInput("DetectorResponse", const_cast<DetectorResponseImageType*>(DetectorResponse));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetMaterialAttenuations(const MaterialAttenuationsImageType *MaterialAttenuations)
{
  this->SetInput("MaterialAttenuations", const_cast<MaterialAttenuationsImageType*>(MaterialAttenuations));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename DecomposedProjectionsType::ConstPointer
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputDecomposedProjections()
{
  return static_cast< const DecomposedProjectionsType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename MeasuredProjectionsType::ConstPointer
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputMeasuredProjections()
{
  return static_cast< const MeasuredProjectionsType * >
          ( this->itk::ProcessObject::GetInput("MeasuredProjections") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename DetectorResponseImageType::ConstPointer
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetDetectorResponse()
{
  return static_cast< const DetectorResponseImageType * >
          ( this->itk::ProcessObject::GetInput("DetectorResponse") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename MaterialAttenuationsImageType::ConstPointer
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetMaterialAttenuations()
{
  return static_cast< const MaterialAttenuationsImageType * >
          ( this->itk::ProcessObject::GetInput("MaterialAttenuations") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
itk::DataObject::Pointer
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::MakeOutput(DataObjectPointerArraySizeType idx)
{
  itk::DataObject::Pointer output;

  switch ( idx )
    {
    case 0:
      output = ( DecomposedProjectionsType::New() ).GetPointer();
      break;
    case 1:
      output = ( DecomposedProjectionsType::New() ).GetPointer();
      break;
    case 2:
      output = ( DecomposedProjectionsType::New() ).GetPointer();
      break;
    }
  return output.GetPointer();
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateInputRequestedRegion()
{
  // Input 0 is the initial decomposed projections
  typename DecomposedProjectionsType::Pointer inputPtr0 =
    const_cast< DecomposedProjectionsType * >( this->GetInputDecomposedProjections().GetPointer() );
  if ( !inputPtr0 ) return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the spectral projections
  typename MeasuredProjectionsType::Pointer inputPtr1 =
    const_cast< MeasuredProjectionsType * >( this->GetInputMeasuredProjections().GetPointer() );
  if ( !inputPtr1 ) return;
  inputPtr1->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 3 is the detector response image (2D float)
  typename DetectorResponseImageType::Pointer inputPtr3 =
    const_cast< DetectorResponseImageType * >( this->GetDetectorResponse().GetPointer() );
  if ( !inputPtr3 ) return;
  inputPtr3->SetRequestedRegion( inputPtr3->GetLargestPossibleRegion() );

  // Input 4 is the material attenuations image (2D float)
  typename MaterialAttenuationsImageType::Pointer inputPtr4 =
    const_cast< MaterialAttenuationsImageType * >( this->GetMaterialAttenuations().GetPointer() );
  if ( !inputPtr4 ) return;
  inputPtr4->SetRequestedRegion( inputPtr4->GetLargestPossibleRegion() );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();
  this->GetOutput(0)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());
  this->GetOutput(1)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());
  this->GetOutput(2)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::BeforeThreadedGenerateData()
{
  // Read the material attenuations image as a matrix
  typename MaterialAttenuationsImageType::IndexType indexMat;
  m_MaterialAttenuations.SetSize(m_NumberOfMaterials, m_NumberOfEnergies);
  for (unsigned int energy=0; energy<m_NumberOfEnergies; energy++)
    {
    indexMat[1] = energy;
    for (unsigned int material=0; material<m_NumberOfMaterials; material++)
      {
      indexMat[0] = material;
      m_MaterialAttenuations[material][energy] = this->GetMaterialAttenuations()->GetPixel(indexMat);
      }
    }

  // If resquested, rescale these attenuations
  if(m_RescaleAttenuations)
    this->RescaleMaterialAttenuations();
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::RescaleMaterialAttenuations() // Rescale the material attenuation functions to match the order of magnitude of the first material's one
{
  m_RescalingFactors.SetSize(m_NumberOfMaterials);
  m_RescalingFactors.Fill(0);
  m_RescalingFactors[0] = 1;

  // Compute the rescaling factors (one per material, except for the first material which serves as a reference)
  // Criterion: the ratio between the attenuation of one material and the attenuation of the first material must
  // have a mean value of one over the range of energies used
  for (unsigned int material=1; material<m_NumberOfMaterials; material++)
    {
    typename MaterialAttenuationsImageType::PixelType nonZeroAttenuationsCount = 0;
    for(unsigned int energy=0; energy<m_NumberOfEnergies; energy++)
      {
      // Make sure both the first and the current material's attenuations are non-zero
      if ((m_MaterialAttenuations[0][energy] != 0) && (m_MaterialAttenuations[material][energy] != 0))
        {
        m_RescalingFactors[material] += m_MaterialAttenuations[material][energy] / m_MaterialAttenuations[0][energy];
        nonZeroAttenuationsCount += 1;
        }
      }
    m_RescalingFactors[material] /= nonZeroAttenuationsCount;
    }

  // Initialize the rescaled material attenuations matrix
  m_RescaledMaterialAttenuations.SetSize(m_NumberOfMaterials, m_NumberOfEnergies);

  // Compute the rescaled attenuations
  for (unsigned int material=0; material<m_NumberOfMaterials; material++)
    for(unsigned int energy=0; energy<m_NumberOfEnergies; energy++)
      m_RescaledMaterialAttenuations[material][energy] = m_MaterialAttenuations[material][energy] / m_RescalingFactors[material];
}


} // end namespace rtk

#endif // rtkSimplexProjectionsDecompositionImageFilter_hxx
