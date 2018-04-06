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

  // Fill in the vectors and matrices with zeros
  m_MaterialAttenuations.fill(0.); //Not sure this works
  m_DetectorResponse.fill(0.);

  // Initial lengths, set to incorrect values to make sure that they are indeed updated
  m_NumberOfSpectralBins = 8;
  m_OutputInverseCramerRaoLowerBound = false;
  m_OutputFischerMatrix = false;
  m_LogTransformEachBin = false;
  m_GuessInitialization = false;
  m_IsSpectralCT = true;
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputDecomposedProjections(const DecomposedProjectionsType* DecomposedProjections)
{
  this->SetNthInput(0, const_cast<DecomposedProjectionsType*>(DecomposedProjections));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputMeasuredProjections(const MeasuredProjectionsType *SpectralProjections)
{
  this->SetInput("MeasuredProjections", const_cast<MeasuredProjectionsType*>(SpectralProjections));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetDetectorResponse(const DetectorResponseImageType *DetectorResponse)
{
  this->SetInput("DetectorResponse", const_cast<DetectorResponseImageType*>(DetectorResponse));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetMaterialAttenuations(const MaterialAttenuationsImageType *MaterialAttenuations)
{
  this->SetInput("MaterialAttenuations", const_cast<MaterialAttenuationsImageType*>(MaterialAttenuations));
}


template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputIncidentSpectrum(const IncidentSpectrumImageType *IncidentSpectrum)
{
  this->SetInput("IncidentSpectrum", const_cast<IncidentSpectrumImageType*>(IncidentSpectrum));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputSecondIncidentSpectrum(const IncidentSpectrumImageType *SecondIncidentSpectrum)
{
  this->SetInput("SecondIncidentSpectrum", const_cast<IncidentSpectrumImageType*>(SecondIncidentSpectrum));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename DecomposedProjectionsType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputDecomposedProjections()
{
  return static_cast< const DecomposedProjectionsType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename MeasuredProjectionsType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputMeasuredProjections()
{
  return static_cast< const MeasuredProjectionsType * >
          ( this->itk::ProcessObject::GetInput("MeasuredProjections") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename DetectorResponseImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetDetectorResponse()
{
  return static_cast< const DetectorResponseImageType * >
          ( this->itk::ProcessObject::GetInput("DetectorResponse") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename MaterialAttenuationsImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetMaterialAttenuations()
{
  return static_cast< const MaterialAttenuationsImageType * >
          ( this->itk::ProcessObject::GetInput("MaterialAttenuations") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputIncidentSpectrum()
{
  return static_cast< const IncidentSpectrumImageType * >
          ( this->itk::ProcessObject::GetInput("IncidentSpectrum") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputSecondIncidentSpectrum()
{
  return static_cast< const IncidentSpectrumImageType * >
          ( this->itk::ProcessObject::GetInput("SecondIncidentSpectrum") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
itk::DataObject::Pointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
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
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();
  this->GetOutput(0)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());
  this->GetOutput(1)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());
  this->GetOutput(2)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());

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


template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  // Read the material attenuations image as a matrix
  typename MaterialAttenuationsImageType::IndexType indexMat;
  this->m_MaterialAttenuations.set_size(this->m_NumberOfEnergies, this->m_NumberOfMaterials);
  for (unsigned int energy=0; energy<this->m_NumberOfEnergies; energy++)
    {
    indexMat[1] = energy;
    for (unsigned int material=0; material<this->m_NumberOfMaterials; material++)
      {
      indexMat[0] = material;
      this->m_MaterialAttenuations[energy][material] = this->GetMaterialAttenuations()->GetPixel(indexMat);
      }
    }

  if (this->GetInputSecondIncidentSpectrum())
    {
    // Read the detector response image as a matrix
    this->m_DetectorResponse.set_size(1, this->m_NumberOfEnergies);
    this->m_DetectorResponse.fill(0);
    typename DetectorResponseImageType::IndexType indexDet;
    for (unsigned int energy=0; energy<this->m_NumberOfEnergies; energy++)
      {
      indexDet[0] = energy;
      indexDet[1] = 0;
      this->m_DetectorResponse[0][energy] += this->GetDetectorResponse()->GetPixel(indexDet);
      }
    }
  else
    {
    // Read the detector response image as a matrix
    this->m_DetectorResponse.set_size(this->m_NumberOfSpectralBins, this->m_NumberOfEnergies);
    this->m_DetectorResponse.fill(0);
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
  rtk::ProjectionsDecompositionNegativeLogLikelihood::Pointer cost;
  if(m_IsSpectralCT)
    cost = rtk::Schlomka2008NegativeLogLikelihood::New();
  else
    cost = rtk::DualEnergyNegativeLogLikelihood::New();

  cost->SetNumberOfEnergies(this->GetNumberOfEnergies());
  cost->SetNumberOfMaterials(this->GetNumberOfMaterials());
  cost->SetNumberOfSpectralBins(this->GetNumberOfSpectralBins());

  // Pass the attenuation functions to the cost function
  cost->SetMaterialAttenuations(this->m_MaterialAttenuations);
  if (m_GuessInitialization && (!this->GetInputSecondIncidentSpectrum()))
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

  // Special case for the dual energy CT
  itk::ImageRegionConstIterator<IncidentSpectrumImageType> secondSpectrumIt;
  if (this->GetInputSecondIncidentSpectrum())
    secondSpectrumIt = itk::ImageRegionConstIterator<IncidentSpectrumImageType>(this->GetInputSecondIncidentSpectrum(), incidentSpectrumRegionForThread);

  while(!output0It.IsAtEnd())
    {
    // The input incident spectrum image typically has lower dimension than
    // This condition makes the iterator cycle over and over on the same image, following the other ones
    if(spectrumIt.IsAtEnd())
      {
      spectrumIt.GoToBegin();
      if (this->GetInputSecondIncidentSpectrum())
        secondSpectrumIt.GoToBegin();
      }

    // Build a vnl_matrix out of the high and low energy incident spectra (if DECT)
    // or out of single spectrum (if spectral)
    vnl_matrix<float> spectra;
    if (this->GetInputSecondIncidentSpectrum()) // Dual energy CT
      {
      spectra.set_size(2, this->m_NumberOfEnergies);
      spectra.set_row(0, spectrumIt.Get().GetDataPointer() );
      spectra.set_row(1, secondSpectrumIt.Get().GetDataPointer() );
      }
    else
      {
      spectra.set_size(1, this->m_NumberOfEnergies);
      spectra.set_row(0, spectrumIt.Get().GetDataPointer() );
      }

    // Pass the incident spectrum vector to cost function
    cost->SetIncidentSpectrum(spectra);
    cost->Initialize();

    // Pass the detector counts vector to cost function
    cost->SetMeasuredData(spectralProjIt.Get());

    // Run the optimizer
    typename rtk::ProjectionsDecompositionNegativeLogLikelihood::ParametersType startingPosition(this->m_NumberOfMaterials);
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
    if (this->GetInputSecondIncidentSpectrum())
      ++secondSpectrumIt;
    }
}

} // end namespace rtk

#endif // rtkSimplexSpectralProjectionsDecompositionImageFilter_hxx
