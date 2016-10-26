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

#ifndef rtkSimplexDualEnergyProjectionsDecompositionImageFilter_hxx
#define rtkSimplexDualEnergyProjectionsDecompositionImageFilter_hxx

#include "rtkSimplexDualEnergyProjectionsDecompositionImageFilter.h"
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

namespace rtk
{

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SimplexDualEnergyProjectionsDecompositionImageFilter()
{
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetHighEnergyIncidentSpectrum(const IncidentSpectrumImageType *IncidentSpectrum)
{
  this->SetInput("HighEnergy", const_cast<IncidentSpectrumImageType*>(IncidentSpectrum));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetHighEnergyIncidentSpectrum()
{
  return static_cast< const IncidentSpectrumImageType * >
          ( this->itk::ProcessObject::GetInput("HighEnergy") );
}


template<typename DecomposedProjectionsType, typename SpectralProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetLowEnergyIncidentSpectrum(const IncidentSpectrumImageType *IncidentSpectrum)
{
  this->SetInput("LowEnergy", const_cast<IncidentSpectrumImageType*>(IncidentSpectrum));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetLowEnergyIncidentSpectrum()
{
  return static_cast< const IncidentSpectrumImageType * >
          ( this->itk::ProcessObject::GetInput("LowEnergy") );
}


template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  this->m_NumberOfMaterials = this->GetInputDecomposedProjections()->GetVectorLength();
  this->m_NumberOfEnergies = this->GetHighEnergyIncidentSpectrum()->GetVectorLength();
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();

  // High energy incident spectrum image (same dimension as a single projection)
  typename IncidentSpectrumImageType::Pointer inputPtrHighEnergySpectrum =
    const_cast< IncidentSpectrumImageType * >( this->GetHighEnergyIncidentSpectrum().GetPointer() );
  if ( !inputPtrHighEnergySpectrum ) return;

  // Low energy incident spectrum image (same dimension as a single projection)
  typename IncidentSpectrumImageType::Pointer inputPtrLowEnergySpectrum =
    const_cast< IncidentSpectrumImageType * >( this->GetLowEnergyIncidentSpectrum().GetPointer() );
  if ( !inputPtrLowEnergySpectrum ) return;

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

  inputPtrHighEnergySpectrum->SetRequestedRegion( requested );
  inputPtrLowEnergySpectrum->SetRequestedRegion( requested );
}


template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  // Read the detector response image as a matrix
  this->m_DetectorResponse.SetSize(1, this->m_NumberOfEnergies);
  this->m_DetectorResponse.Fill(0);
  typename DetectorResponseImageType::IndexType indexDet;
  for (unsigned int energy=0; energy<this->m_NumberOfEnergies; energy++)
    {
    indexDet[0] = energy;
    indexDet[1] = 0;
    this->m_DetectorResponse[0][energy] += this->GetDetectorResponse()->GetPixel(indexDet);
    }
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexDualEnergyProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::ThreadedGenerateData(const typename DecomposedProjectionsType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  ////////////////////////////////////////////////////////////////////
  // Create a Nelder-Mead simplex optimizer and its cost function
  itk::AmoebaOptimizer::Pointer optimizer = itk::AmoebaOptimizer::New();
  typename CostFunctionType::Pointer cost = CostFunctionType::New();

  cost->SetNumberOfEnergies(this->GetNumberOfEnergies());
  cost->SetNumberOfMaterials(this->GetNumberOfMaterials());

  // Pass the attenuation functions to the cost function
  cost->SetMaterialAttenuations(this->m_MaterialAttenuations);

  // Pass the binned detector response to the cost function
  cost->SetDetectorResponse(this->m_DetectorResponse);

  // Set the optimizer
  optimizer->SetCostFunction(cost);
  optimizer->SetMaximumNumberOfIterations(this->m_NumberOfIterations);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Walk the output projection stack. For each pixel, set the cost function's member variables and run the optimizer.
  itk::ImageRegionIterator<DecomposedProjectionsType> output0It (this->GetOutput(0), outputRegionForThread);
  itk::ImageRegionIterator<DecomposedProjectionsType> output1It (this->GetOutput(1), outputRegionForThread);
  itk::ImageRegionConstIterator<DecomposedProjectionsType> inputIt (this->GetInputDecomposedProjections(), outputRegionForThread);
  itk::ImageRegionConstIterator<MeasuredProjectionsType> measuredDataIt (this->GetInputMeasuredProjections(), outputRegionForThread);

  typename IncidentSpectrumImageType::RegionType incidentSpectrumRegionForThread;
  for (unsigned int dim=0; dim<IncidentSpectrumImageType::GetImageDimension(); dim++)
    {
    incidentSpectrumRegionForThread.SetIndex(dim, outputRegionForThread.GetIndex()[dim]);
    incidentSpectrumRegionForThread.SetSize(dim, outputRegionForThread.GetSize()[dim]);
    }
  itk::ImageRegionConstIterator<IncidentSpectrumImageType> highEnergySpectrumIt (this->GetHighEnergyIncidentSpectrum(), incidentSpectrumRegionForThread);
  itk::ImageRegionConstIterator<IncidentSpectrumImageType> lowEnergySpectrumIt (this->GetLowEnergyIncidentSpectrum(), incidentSpectrumRegionForThread);

  while(!output0It.IsAtEnd())
    {
    // The input incident spectrum image typically has lower dimension than
    // This condition makes the iterator cycle over and over on the same image, following the other ones
    if(highEnergySpectrumIt.IsAtEnd())
      {
      highEnergySpectrumIt.GoToBegin();
      lowEnergySpectrumIt.GoToBegin();
      }

    // Pass the incident spectrum vector to cost function
    cost->SetHighEnergyIncidentSpectrum(highEnergySpectrumIt.Get());
    cost->SetLowEnergyIncidentSpectrum(lowEnergySpectrumIt.Get());

    // Pass the detector counts vector to cost function
    cost->SetMeasuredData(measuredDataIt.Get());

    // Run the optimizer
    typename CostFunctionType::ParametersType startingPosition(this->m_NumberOfMaterials);
    for (unsigned int m=0; m<this->m_NumberOfMaterials; m++)
      startingPosition[m] = inputIt.Get()[m];

    optimizer->SetInitialPosition(startingPosition);
    optimizer->SetAutomaticInitialSimplex(true);
    optimizer->SetOptimizeWithRestarts(this->m_OptimizeWithRestarts);
    optimizer->StartOptimization();

    typename DecomposedProjectionsType::PixelType outputPixel;
    outputPixel.SetSize(this->m_NumberOfMaterials);
    for (unsigned int m=0; m<this->m_NumberOfMaterials; m++)
      outputPixel[m] = optimizer->GetCurrentPosition()[m];
    output0It.Set(outputPixel);

    // Compute the inverse variance of decomposition noise, and store it into output(1)
    output1It.Set(cost->GetInverseCramerRaoLowerBound(optimizer->GetCurrentPosition()));

    // Move forward
    ++output0It;
    ++output1It;
    ++inputIt;
    ++measuredDataIt;
    ++highEnergySpectrumIt;
    ++lowEnergySpectrumIt;
    }
}

} // end namespace rtk

#endif // rtkSimplexDualEnergyProjectionsDecompositionImageFilter_hxx
