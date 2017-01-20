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

#ifndef rtkSpectralForwardModelImageFilter_hxx
#define rtkSpectralForwardModelImageFilter_hxx

#include "rtkSpectralForwardModelImageFilter.h"
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

namespace rtk
{

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SpectralForwardModelImageFilter()
{
  // Initial lengths, set to incorrect values to make sure that they are indeed updated
  m_NumberOfSpectralBins = 8;
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputMeasuredProjections(const MeasuredProjectionsType *SpectralProjections)
{
  this->SetNthInput(0, const_cast<MeasuredProjectionsType*>(SpectralProjections));
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType, SpectralProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputIncidentSpectrum(const IncidentSpectrumImageType *IncidentSpectrum)
{
  this->SetInput("IncidentSpectrum", const_cast<IncidentSpectrumImageType*>(IncidentSpectrum));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetInputDecomposedProjections(const DecomposedProjectionsType* DecomposedProjections)
{
  this->SetInput("DecomposedProjections", const_cast<DecomposedProjectionsType*>(DecomposedProjections));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetDetectorResponse(const DetectorResponseImageType *DetectorResponse)
{
  this->SetInput("DetectorResponse", const_cast<DetectorResponseImageType*>(DetectorResponse));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::SetMaterialAttenuations(const MaterialAttenuationsImageType *MaterialAttenuations)
{
  this->SetInput("MaterialAttenuations", const_cast<MaterialAttenuationsImageType*>(MaterialAttenuations));
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename MeasuredProjectionsType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputMeasuredProjections()
{
  return static_cast< const MeasuredProjectionsType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template<typename DecomposedProjectionsType, typename SpectralProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType, SpectralProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputIncidentSpectrum()
{
  return static_cast< const IncidentSpectrumImageType * >
          ( this->itk::ProcessObject::GetInput("IncidentSpectrum") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename DecomposedProjectionsType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetInputDecomposedProjections()
{
  return static_cast< const DecomposedProjectionsType * >
          ( this->itk::ProcessObject::GetInput("DecomposedProjections") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename DetectorResponseImageType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetDetectorResponse()
{
  return static_cast< const DetectorResponseImageType * >
          ( this->itk::ProcessObject::GetInput("DetectorResponse") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
typename MaterialAttenuationsImageType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GetMaterialAttenuations()
{
  return static_cast< const MaterialAttenuationsImageType * >
          ( this->itk::ProcessObject::GetInput("MaterialAttenuations") );
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  this->m_NumberOfSpectralBins = this->GetInputMeasuredProjections()->GetVectorLength();
  this->m_NumberOfMaterials = this->GetInputDecomposedProjections()->GetVectorLength();
  this->m_NumberOfEnergies = this->GetInputIncidentSpectrum()->GetVectorLength();
}

template<typename DecomposedProjectionsType, typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType, typename DetectorResponseImageType, typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
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
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

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
SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType,
                                                   IncidentSpectrumImageType, DetectorResponseImageType, MaterialAttenuationsImageType>
::ThreadedGenerateData(const typename OutputImageType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  ////////////////////////////////////////////////////////////////////
  // Create a Nelder-Mead simplex optimizer and its cost function
  typename CostFunctionType::Pointer cost = CostFunctionType::New();

  cost->SetNumberOfEnergies(this->GetNumberOfEnergies());
  cost->SetNumberOfMaterials(this->GetNumberOfMaterials());
  cost->SetNumberOfSpectralBins(this->GetNumberOfSpectralBins());

  // Pass the attenuation functions to the cost function
  cost->SetMaterialAttenuations(this->m_MaterialAttenuations);

  // Pass the binned detector response to the cost function
  cost->SetDetectorResponse(this->m_DetectorResponse);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Walk the output projection stack. For each pixel, set the cost function's member variables and run the optimizer.
  itk::ImageRegionIterator<MeasuredProjectionsType> outIt (this->GetOutput(0), outputRegionForThread);
  itk::ImageRegionConstIterator<DecomposedProjectionsType> inIt (this->GetInputDecomposedProjections(), outputRegionForThread);

  typename IncidentSpectrumImageType::RegionType incidentSpectrumRegionForThread;
  for (unsigned int dim=0; dim<IncidentSpectrumImageType::GetImageDimension(); dim++)
    {
    incidentSpectrumRegionForThread.SetIndex(dim, outputRegionForThread.GetIndex()[dim]);
    incidentSpectrumRegionForThread.SetSize(dim, outputRegionForThread.GetSize()[dim]);
    }
  itk::ImageRegionConstIterator<IncidentSpectrumImageType> spectrumIt (this->GetInputIncidentSpectrum(), incidentSpectrumRegionForThread);

  while(!outIt.IsAtEnd())
    {
    // The input incident spectrum image typically has lower dimension than decomposed projections
    // This condition makes the iterator cycle over and over on the same image, following the other ones
    if(spectrumIt.IsAtEnd())
      spectrumIt.GoToBegin();

    // Pass the incident spectrum vector to cost function
    cost->SetIncidentSpectrum(spectrumIt.Get());

    // Run the optimizer
    typename CostFunctionType::ParametersType in(this->m_NumberOfMaterials);
    for (unsigned int m=0; m<this->m_NumberOfMaterials; m++)
      in[m] = inIt.Get()[m];

    // Compute the expected forward projection
    vnl_vector<double> forward = cost->ForwardModel(in);

    // Fill a fixed-sized vector with the computed photon counts in each bin
    typename MeasuredProjectionsType::PixelType outputPixel;
    outputPixel.SetSize(this->m_NumberOfSpectralBins);
    for (unsigned int m=0; m<this->m_NumberOfSpectralBins; m++)
      outputPixel[m] = forward[m];

    // Write it to the output
    outIt.Set(outputPixel);

    // Move forward
    ++outIt;
    ++inIt;
    ++spectrumIt;
    }
}

} // end namespace rtk

#endif // rtkSpectralForwardModelImageFilter_hxx
