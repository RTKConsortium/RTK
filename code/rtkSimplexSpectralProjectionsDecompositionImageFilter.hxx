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

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SimplexSpectralProjectionsDecompositionImageFilter()
{
  // Initial lengths, set to incorrect values to make sure that they are indeed updated
  m_NumberOfSpectralBins = 8;
  m_OutputInverseCramerRaoLowerBound = false;
  m_OutputFischerMatrix = false;
  m_LogTransformEachBin = false;
  m_GuessInitialization = false;
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputIncidentSpectrum(const IncidentSpectrumImageType *IncidentSpectrum)
{
  this->SetNthInput(2, const_cast<IncidentSpectrumImageType*>(IncidentSpectrum));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, SpectralProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputIncidentSpectrum()
{
  return static_cast< const IncidentSpectrumImageType * >
          ( this->itk::ProcessObject::GetInput(2) );
}


template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  this->m_NumberOfSpectralBins = this->GetInputMeasuredProjections()->GetVectorLength();
  this->m_NumberOfMaterials = this->GetInputDecomposedProjections()->GetVectorLength();
  this->m_NumberOfEnergies = this->GetInputIncidentSpectrum()->GetVectorLength();

  // Set vector length for the fischer matrix
  this->GetOutput(2)->SetVectorLength(this->m_NumberOfMaterials * this->m_NumberOfMaterials);

  // Change vector length for the decomposed projections, if required
  if (m_LogTransformEachBin)
    this->GetOutput(0)->SetVectorLength(this->m_NumberOfMaterials + this->m_NumberOfSpectralBins);
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();

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
}


template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  // Read the detector response image as a matrix
  this->m_DetectorResponse.SetSize(this->m_NumberOfSpectralBins, this->m_NumberOfEnergies);
  this->m_DetectorResponse.Fill(0);
  typename DetectorResponseImageType::IndexType indexDet;
  for (unsigned int energy=0; energy<this->m_NumberOfEnergies; energy++)
    {
    indexDet[0] = energy;
    for (unsigned int bin=0; bin<m_NumberOfSpectralBins; bin++)
      {
      for (unsigned int pulseHeight=m_Thresholds[bin]-1; pulseHeight<m_Thresholds[bin+1]; pulseHeight++)
        {
        indexDet[1] = pulseHeight;
        // Linear interpolation on the pulse heights: half of the pulses that have "threshold"
        // height are considered below threshold, the other half are considered above threshold
        if ((pulseHeight == m_Thresholds[bin]-1) || (pulseHeight == m_Thresholds[bin+1] - 1))
          this->m_DetectorResponse[bin][energy] += this->GetDetectorResponse()->GetPixel(indexDet) / 2;
        else
          this->m_DetectorResponse[bin][energy] += this->GetDetectorResponse()->GetPixel(indexDet);
        }
      }
    }
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::ThreadedGenerateData(const typename DecomposedProjectionsType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  ////////////////////////////////////////////////////////////////////
  // Create a Nelder-Mead simplex optimizer and its cost function
  itk::AmoebaOptimizer::Pointer optimizer = itk::AmoebaOptimizer::New();
  typename CostFunctionType::Pointer cost = CostFunctionType::New();

  cost->SetNumberOfEnergies(this->GetNumberOfEnergies());
  cost->SetNumberOfMaterials(this->GetNumberOfMaterials());
  cost->SetNumberOfSpectralBins(this->GetNumberOfSpectralBins());

  // Pass the attenuation functions to the cost function
  if (this->m_RescaleAttenuations)
    cost->SetMaterialAttenuations(this->m_RescaledMaterialAttenuations);
  else
    cost->SetMaterialAttenuations(this->m_MaterialAttenuations);
  if (m_GuessInitialization)
    cost->SetThresholds(this->m_Thresholds);

  // Pass the binned detector response to the cost function
  cost->SetDetectorResponse(this->m_DetectorResponse);

  // Set the optimizer
  optimizer->SetCostFunction(cost);
  optimizer->SetMaximumNumberOfIterations(this->m_NumberOfIterations);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Walk the output projection stack. For each pixel, set the cost function's member variables and run the optimizer.
  itk::ImageRegionIterator<DecomposedProjectionsType> output0It (this->GetOutput(0), outputRegionForThread);
  itk::ImageRegionIterator<DecomposedProjectionsType> output1It (this->GetOutput(1), outputRegionForThread);
  itk::ImageRegionIterator<DecomposedProjectionsType> output2It (this->GetOutput(2), outputRegionForThread);
  itk::ImageRegionConstIterator<DecomposedProjectionsType> inputIt (this->GetInputDecomposedProjections(), outputRegionForThread);
  itk::ImageRegionConstIterator<MeasuredProjectionsType> spectralProjIt (this->GetInputMeasuredProjections(), outputRegionForThread);

  typename IncidentSpectrumImageType::RegionType incidentSpectrumRegionForThread;
  for (unsigned int dim=0; dim<IncidentSpectrumImageType::GetImageDimension(); dim++)
    {
    incidentSpectrumRegionForThread.SetIndex(dim, outputRegionForThread.GetIndex()[dim]);
    incidentSpectrumRegionForThread.SetSize(dim, outputRegionForThread.GetSize()[dim]);
    }
  itk::ImageRegionConstIterator<IncidentSpectrumImageType> spectrumIt (this->GetInputIncidentSpectrum(), incidentSpectrumRegionForThread);

  while(!output0It.IsAtEnd())
    {
    // The input incident spectrum image typically has lower dimension than
    // This condition makes the iterator cycle over and over on the same image, following the other ones
    if(spectrumIt.IsAtEnd())
      spectrumIt.GoToBegin();

    // Pass the incident spectrum vector to cost function
    cost->SetIncidentSpectrum(spectrumIt.Get());

    // Pass the detector counts vector to cost function
    cost->SetMeasuredData(spectralProjIt.Get());


    // Run the optimizer
    typename CostFunctionType::ParametersType startingPosition(this->m_NumberOfMaterials);
    if (m_GuessInitialization)
      {
      itk::VariableLengthVector<double> guess = cost->GuessInitialization();
      for (unsigned int m=0; m<this->m_NumberOfMaterials; m++)
        startingPosition[m] = guess[m];
      }
    else
      {
      for (unsigned int m=0; m<this->m_NumberOfMaterials; m++)
        startingPosition[m] = inputIt.Get()[m];
      }

    optimizer->SetInitialPosition(startingPosition);
    optimizer->SetAutomaticInitialSimplex(true);
    optimizer->SetOptimizeWithRestarts(this->m_OptimizeWithRestarts);
    optimizer->StartOptimization();

    typename DecomposedProjectionsType::PixelType outputPixel;
    if (m_LogTransformEachBin)
      {
      outputPixel.SetSize(this->m_NumberOfMaterials + this->m_NumberOfSpectralBins);
      for (unsigned int bin=0; bin<this->m_NumberOfSpectralBins; bin++)
        outputPixel[bin+this->m_NumberOfMaterials] = cost->BinwiseLogTransform()[bin];
      }
    else
      outputPixel.SetSize(this->m_NumberOfMaterials);

    for (unsigned int m=0; m<this->m_NumberOfMaterials; m++)
      outputPixel[m] = optimizer->GetCurrentPosition()[m];

    output0It.Set(outputPixel);

    // If required, compute the Fischer matrix
    if (m_OutputInverseCramerRaoLowerBound || m_OutputFischerMatrix)
      cost->ComputeFischerMatrix(optimizer->GetCurrentPosition());

    // If requested, compute the inverse variance of decomposition noise, and store it into output(1)
    if (m_OutputInverseCramerRaoLowerBound)
      output1It.Set(cost->GetInverseCramerRaoLowerBound());

    // If requested, store the Fischer matrix into output(2)
    if (m_OutputFischerMatrix)
      output2It.Set(cost->GetFischerMatrix());

    // Move forward
    ++output0It;
    ++output1It;
    ++output2It;
    ++inputIt;
    ++spectralProjIt;
    ++spectrumIt;
    }
}

} // end namespace rtk

#endif // rtkSimplexSpectralProjectionsDecompositionImageFilter_hxx
