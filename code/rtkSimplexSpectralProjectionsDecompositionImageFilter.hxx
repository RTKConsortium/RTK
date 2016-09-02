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

#ifndef rtkSimplexSpectralProjectionsDecompositionImageFilter_hxx
#define rtkSimplexSpectralProjectionsDecompositionImageFilter_hxx

#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

namespace rtk
{

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SimplexSpectralProjectionsDecompositionImageFilter()
{
  this->SetNumberOfRequiredInputs(5); // decomposed projections, photon counts, incident spectrum, detector response, material attenuations

  // Set the default values of member parameters
  m_NumberOfIterations=300;

  // Fill in the vectors and matrices with zeros
  m_MaterialAttenuations.Fill(0.); //Not sure this works
  m_DetectorResponse.Fill(0.);
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputDecomposedProjections(const DecomposedProjectionsType* DecomposedProjections)
{
  this->SetNthInput(0, const_cast<DecomposedProjectionsType*>(DecomposedProjections));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputSpectralProjections(const SpectralProjectionsType *SpectralProjections)
{
  this->SetNthInput(1, const_cast<SpectralProjectionsType*>(SpectralProjections));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputIncidentSpectrum(const IncidentSpectrumImageType *IncidentSpectrum)
{
  this->SetNthInput(2, const_cast<IncidentSpectrumImageType*>(IncidentSpectrum));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetDetectorResponse(const DetectorResponseImageType *DetectorResponse)
{
  this->SetNthInput(3, const_cast<DetectorResponseImageType*>(DetectorResponse));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetMaterialAttenuations(const MaterialAttenuationsImageType *MaterialAttenuations)
{
  this->SetNthInput(4, const_cast<MaterialAttenuationsImageType*>(MaterialAttenuations));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename DecomposedProjectionsType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputDecomposedProjections()
{
  return static_cast< const DecomposedProjectionsType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename SpectralProjectionsType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputSpectralProjections()
{
  return static_cast< const SpectralProjectionsType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputIncidentSpectrum()
{
  return static_cast< const IncidentSpectrumImageType * >
          ( this->itk::ProcessObject::GetInput(2) );
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename DetectorResponseImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetDetectorResponse()
{
  return static_cast< const DetectorResponseImageType * >
          ( this->itk::ProcessObject::GetInput(3) );
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename MaterialAttenuationsImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetMaterialAttenuations()
{
  return static_cast< const MaterialAttenuationsImageType * >
          ( this->itk::ProcessObject::GetInput(4) );
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateInputRequestedRegion()
{
  // Input 0 is the initial decomposed projections
  typename DecomposedProjectionsType::Pointer inputPtr0 =
    const_cast< DecomposedProjectionsType * >( this->GetInputDecomposedProjections().GetPointer() );
  if ( !inputPtr0 ) return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the spectral projections
  typename SpectralProjectionsType::Pointer inputPtr1 =
    const_cast< SpectralProjectionsType * >( this->GetInputSpectralProjections().GetPointer() );
  if ( !inputPtr1 ) return;
  inputPtr1->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 2 is the incident spectrum image (same dimension as a single projection)
  typename IncidentSpectrumImageType::Pointer inputPtr2 =
    const_cast< IncidentSpectrumImageType * >( this->GetInputIncidentSpectrum().GetPointer() );
  if ( !inputPtr2 ) return;

  typename IncidentSpectrumImageType::RegionType requested;
  typename IncidentSpectrumImageType::IndexType indexRequested;
  typename IncidentSpectrumImageType::SizeType sizeRequested;
  indexRequested.Fill(0);
  sizeRequested.Fill(0);
  for (unsigned int i=0; i< IncidentSpectrumImageType::GetImageDimension(); i++)
    {
    indexRequested[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    sizeRequested[i] = this->GetOutput()->GetRequestedRegion().GetSize()[i];
    }
  requested.SetIndex(indexRequested);
  requested.SetSize(sizeRequested);

  inputPtr2->SetRequestedRegion( requested );

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

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();
  this->GetOutput()->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::BeforeThreadedGenerateData()
{
  unsigned int NumberOfSpectralBins = SpectralProjectionsType::PixelType::Dimension;
  unsigned int NumberOfMaterials = DecomposedProjectionsType::PixelType::Dimension;

  // Format the inputs to pass them to the cost function
  // First the detector response matrix
  m_DetectorResponse.Fill(0);
  typename DetectorResponseImageType::IndexType indexDet;
  for (unsigned int energy=0; energy<NumberOfEnergies; energy++)
    {
    indexDet[0] = energy;
    for (unsigned int bin=0; bin<NumberOfSpectralBins; bin++)
      {
      for (unsigned int pulseHeight=m_Thresholds[bin]-1; pulseHeight<m_Thresholds[bin+1]-1; pulseHeight++)
        {
        indexDet[1] = pulseHeight;
        m_DetectorResponse[bin][energy] += this->GetDetectorResponse()->GetPixel(indexDet);
        }
      }
    }

  // Then the material attenuations vector of vectors
  typename MaterialAttenuationsImageType::IndexType indexMat;
  for (unsigned int energy=0; energy<NumberOfEnergies; energy++)
    {
    indexMat[1] = energy;
    for (unsigned int material=0; material<NumberOfMaterials; material++)
      {
      indexMat[0] = material;
      m_MaterialAttenuations[material][energy] = this->GetMaterialAttenuations()->GetPixel(indexMat);
      }
    }
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType, unsigned int NumberOfEnergies,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType, NumberOfEnergies,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::ThreadedGenerateData(const typename DecomposedProjectionsType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  const unsigned int NumberOfMaterials = DecomposedProjectionsType::PixelType::Dimension;

  ////////////////////////////////////////////////////////////////////
  // Create a Nelder-Mead simplex optimizer and its cost function
  itk::AmoebaOptimizer::Pointer optimizer = itk::AmoebaOptimizer::New();
  typename CostFunctionType::Pointer cost = CostFunctionType::New();

  // Pass the attenuation functions to the cost function
  cost->SetMaterialAttenuations(this->m_MaterialAttenuations);

  // Pass the binned detector response to the cost function
  cost->SetDetectorResponse(this->m_DetectorResponse);

  // Set the optimizer
  optimizer->SetCostFunction(cost);
  optimizer->SetMaximumNumberOfIterations(this->m_NumberOfIterations);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Walk the output projection stack. For each pixel, set the cost function's member variables and run the optimizer.
  itk::ImageRegionIterator<DecomposedProjectionsType> outputIt (this->GetOutput(), outputRegionForThread);
  itk::ImageRegionConstIterator<DecomposedProjectionsType> inputIt (this->GetInputDecomposedProjections(), outputRegionForThread);
  itk::ImageRegionConstIterator<SpectralProjectionsType> spectralProjIt (this->GetInputSpectralProjections(), outputRegionForThread);

  typename IncidentSpectrumImageType::RegionType incidentSpectrumRegionForThread;
  for (unsigned int dim=0; dim<IncidentSpectrumImageType::GetImageDimension(); dim++)
    {
    incidentSpectrumRegionForThread.SetIndex(dim, outputRegionForThread.GetIndex()[dim]);
    incidentSpectrumRegionForThread.SetSize(dim, outputRegionForThread.GetSize()[dim]);
    }
  itk::ImageRegionConstIterator<IncidentSpectrumImageType> spectrumIt (this->GetInputIncidentSpectrum(), incidentSpectrumRegionForThread);

  while(!outputIt.IsAtEnd())
    {
    // The input incident spectrum image typically has lower dimension than
    // This condition makes the iterator cycle over and over on the same image, following the other ones
    if(spectrumIt.IsAtEnd())
      spectrumIt.GoToBegin();

    // Pass the incident spectrum vector to cost function
    cost->SetIncidentSpectrum(spectrumIt.Get());

    // Pass the detector counts vector to cost function
    cost->SetDetectorCounts(spectralProjIt.Get());

    // Run the optimizer
    typename CostFunctionType::ParametersType startingPosition(NumberOfMaterials);
    for (unsigned int m=0; m<NumberOfMaterials; m++)
      startingPosition[m] = inputIt.Get()[m];

    optimizer->SetInitialPosition(startingPosition);
    optimizer->SetAutomaticInitialSimplex(true);
    optimizer->SetOptimizeWithRestarts(true);
    srand(0);
    optimizer->StartOptimization();

    typename DecomposedProjectionsType::PixelType outputPixel;
    for (unsigned int m=0; m<NumberOfMaterials; m++)
      outputPixel[m] = optimizer->GetCurrentPosition()[m];

    outputIt.Set(outputPixel);

    // Move forward
    ++outputIt;
    ++inputIt;
    ++spectralProjIt;
    ++spectrumIt;
    }
}

} // end namespace rtk

#endif // rtkSimplexSpectralProjectionsDecompositionImageFilter_hxx
