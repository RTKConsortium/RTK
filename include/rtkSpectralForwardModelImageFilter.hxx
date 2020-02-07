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

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::SpectralForwardModelImageFilter()
{
  // Initial lengths, set to incorrect values to make sure that they are indeed updated
  m_NumberOfSpectralBins = 8;
  m_IsSpectralCT = true;
  m_ComputeVariances = false;

  this->SetNumberOfIndexedOutputs(2); // decomposed projections, inverse variance of decomposition noise

  // Dual energy projections (mean of the distribution)
  this->SetNthOutput(0, this->MakeOutput(0));

  // Variance of the distribution
  this->SetNthOutput(1, this->MakeOutput(1));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<
  DecomposedProjectionsType,
  MeasuredProjectionsType,
  IncidentSpectrumImageType,
  DetectorResponseImageType,
  MaterialAttenuationsImageType>::SetInputMeasuredProjections(const MeasuredProjectionsType * SpectralProjections)
{
  this->SetNthInput(0, const_cast<MeasuredProjectionsType *>(SpectralProjections));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<
  DecomposedProjectionsType,
  MeasuredProjectionsType,
  IncidentSpectrumImageType,
  DetectorResponseImageType,
  MaterialAttenuationsImageType>::SetInputIncidentSpectrum(const IncidentSpectrumImageType * IncidentSpectrum)
{
  this->SetInput("IncidentSpectrum", const_cast<IncidentSpectrumImageType *>(IncidentSpectrum));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::
  SetInputSecondIncidentSpectrum(const IncidentSpectrumImageType * SecondIncidentSpectrum)
{
  this->SetInput("SecondIncidentSpectrum", const_cast<IncidentSpectrumImageType *>(SecondIncidentSpectrum));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<
  DecomposedProjectionsType,
  MeasuredProjectionsType,
  IncidentSpectrumImageType,
  DetectorResponseImageType,
  MaterialAttenuationsImageType>::SetInputDecomposedProjections(const DecomposedProjectionsType * DecomposedProjections)
{
  this->SetInput("DecomposedProjections", const_cast<DecomposedProjectionsType *>(DecomposedProjections));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::SetDetectorResponse(const DetectorResponseImageType *
                                                                                      DetectorResponse)
{
  this->SetInput("DetectorResponse", const_cast<DetectorResponseImageType *>(DetectorResponse));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<
  DecomposedProjectionsType,
  MeasuredProjectionsType,
  IncidentSpectrumImageType,
  DetectorResponseImageType,
  MaterialAttenuationsImageType>::SetMaterialAttenuations(const MaterialAttenuationsImageType * MaterialAttenuations)
{
  this->SetInput("MaterialAttenuations", const_cast<MaterialAttenuationsImageType *>(MaterialAttenuations));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename MeasuredProjectionsType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::GetInputMeasuredProjections()
{
  return static_cast<const MeasuredProjectionsType *>(this->itk::ProcessObject::GetInput(0));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::GetInputIncidentSpectrum()
{
  return static_cast<const IncidentSpectrumImageType *>(this->itk::ProcessObject::GetInput("IncidentSpectrum"));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename IncidentSpectrumImageType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::GetInputSecondIncidentSpectrum()
{
  return static_cast<const IncidentSpectrumImageType *>(this->itk::ProcessObject::GetInput("SecondIncidentSpectrum"));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename DecomposedProjectionsType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::GetInputDecomposedProjections()
{
  return static_cast<const DecomposedProjectionsType *>(this->itk::ProcessObject::GetInput("DecomposedProjections"));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename DetectorResponseImageType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::GetDetectorResponse()
{
  return static_cast<const DetectorResponseImageType *>(this->itk::ProcessObject::GetInput("DetectorResponse"));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename MaterialAttenuationsImageType::ConstPointer
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::GetMaterialAttenuations()
{
  return static_cast<const MaterialAttenuationsImageType *>(this->itk::ProcessObject::GetInput("MaterialAttenuations"));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
itk::DataObject::Pointer
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::MakeOutput(DataObjectPointerArraySizeType idx)
{
  itk::DataObject::Pointer output;

  switch (idx)
  {
    case 0:
      output = (DecomposedProjectionsType::New()).GetPointer();
      break;
    case 1:
      output = (DecomposedProjectionsType::New()).GetPointer();
      break;
  }
  return output.GetPointer();
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  this->m_NumberOfSpectralBins = this->GetInputMeasuredProjections()->GetVectorLength();
  this->m_NumberOfMaterials = this->GetInputDecomposedProjections()->GetVectorLength();
  this->m_NumberOfEnergies = this->GetInputIncidentSpectrum()->GetVectorLength();
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();

  // Input 2 is the incident spectrum image (same dimension as a single projection)
  typename IncidentSpectrumImageType::Pointer inputPtr2 =
    const_cast<IncidentSpectrumImageType *>(this->GetInputIncidentSpectrum().GetPointer());
  if (!inputPtr2)
    return;

  typename IncidentSpectrumImageType::RegionType requested =
    this->GetInputIncidentSpectrum()->GetLargestPossibleRegion();
  typename IncidentSpectrumImageType::IndexType indexRequested;
  typename IncidentSpectrumImageType::SizeType  sizeRequested;
  indexRequested.Fill(0);
  sizeRequested.Fill(0);
  for (unsigned int i = 0; i < IncidentSpectrumImageType::GetImageDimension() - 1; i++)
  {
    indexRequested[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    sizeRequested[i] = this->GetOutput()->GetRequestedRegion().GetSize()[i];
  }

  requested.SetIndex(indexRequested);
  requested.SetSize(sizeRequested);

  inputPtr2->SetRequestedRegion(requested);
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  // Read the material attenuations image as a matrix
  typename MaterialAttenuationsImageType::IndexType indexMat;
  this->m_MaterialAttenuations.set_size(m_NumberOfEnergies, m_NumberOfMaterials);
  for (unsigned int energy = 0; energy < m_NumberOfEnergies; energy++)
  {
    indexMat[1] = energy;
    for (unsigned int material = 0; material < m_NumberOfMaterials; material++)
    {
      indexMat[0] = material;
      m_MaterialAttenuations[energy][material] = this->GetMaterialAttenuations()->GetPixel(indexMat);
    }
  }

  if (this->GetInputSecondIncidentSpectrum())
  {
    // Read the detector response image as a matrix
    this->m_DetectorResponse.set_size(1, this->m_NumberOfEnergies);
    this->m_DetectorResponse.fill(0);
    typename DetectorResponseImageType::IndexType indexDet;
    for (unsigned int energy = 0; energy < this->m_NumberOfEnergies; energy++)
    {
      indexDet[0] = energy;
      indexDet[1] = 0;
      this->m_DetectorResponse[0][energy] += this->GetDetectorResponse()->GetPixel(indexDet);
    }
  }
  else
  {
    this->m_DetectorResponse = SpectralBinDetectorResponse<DetectorResponseType::element_type>(
      this->GetDetectorResponse().GetPointer(), m_Thresholds, m_NumberOfEnergies);
  }
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SpectralForwardModelImageFilter<DecomposedProjectionsType,
                                MeasuredProjectionsType,
                                IncidentSpectrumImageType,
                                DetectorResponseImageType,
                                MaterialAttenuationsImageType>::
  DynamicThreadedGenerateData(const typename OutputImageType::RegionType & outputRegionForThread)
{
  ////////////////////////////////////////////////////////////////////
  // Create a Nelder-Mead simplex optimizer and its cost function
  rtk::ProjectionsDecompositionNegativeLogLikelihood::Pointer cost;
  if (m_IsSpectralCT)
    cost = rtk::Schlomka2008NegativeLogLikelihood::New();
  else
    cost = rtk::DualEnergyNegativeLogLikelihood::New();

  cost->SetNumberOfEnergies(this->GetNumberOfEnergies());
  cost->SetNumberOfMaterials(this->GetNumberOfMaterials());
  cost->SetNumberOfSpectralBins(this->GetNumberOfSpectralBins());

  // Pass the attenuation functions to the cost function
  cost->SetMaterialAttenuations(this->m_MaterialAttenuations);

  // Pass the binned detector response to the cost function
  cost->SetDetectorResponse(this->m_DetectorResponse);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Walk the output projection stack. For each pixel, set the cost function's member variables and run the optimizer.
  itk::ImageRegionIterator<MeasuredProjectionsType>        output0It(this->GetOutput(0), outputRegionForThread);
  itk::ImageRegionIterator<DecomposedProjectionsType>      output1It(this->GetOutput(1), outputRegionForThread);
  itk::ImageRegionConstIterator<DecomposedProjectionsType> inIt(this->GetInputDecomposedProjections(),
                                                                outputRegionForThread);

  typename IncidentSpectrumImageType::RegionType incidentSpectrumRegionForThread =
    this->GetInputIncidentSpectrum()->GetLargestPossibleRegion();
  for (unsigned int dim = 0; dim < DecomposedProjectionsType::GetImageDimension() - 1; dim++)
  {
    incidentSpectrumRegionForThread.SetIndex(dim, outputRegionForThread.GetIndex()[dim]);
    incidentSpectrumRegionForThread.SetSize(dim, outputRegionForThread.GetSize()[dim]);
  }
  itk::ImageRegionConstIterator<IncidentSpectrumImageType> spectrumIt(this->GetInputIncidentSpectrum(),
                                                                      incidentSpectrumRegionForThread);

  // Special case for the dual energy CT
  itk::ImageRegionConstIterator<IncidentSpectrumImageType> secondSpectrumIt;
  if (this->GetInputSecondIncidentSpectrum())
    secondSpectrumIt = itk::ImageRegionConstIterator<IncidentSpectrumImageType>(this->GetInputSecondIncidentSpectrum(),
                                                                                incidentSpectrumRegionForThread);

  while (!output0It.IsAtEnd())
  {
    // The input incident spectrum image typically has lower dimension than decomposed projections
    // This condition makes the iterator cycle over and over on the same image, following the other ones
    if (spectrumIt.IsAtEnd())
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
      spectra.set_size(2, m_NumberOfEnergies);
      spectra.set_row(0, spectrumIt.Get().GetDataPointer());
      spectra.set_row(1, secondSpectrumIt.Get().GetDataPointer());
    }
    else
    {
      spectra.set_size(1, m_NumberOfEnergies);
      spectra.set_row(0, spectrumIt.Get().GetDataPointer());
    }

    // Pass the incident spectrum vector to cost function
    cost->SetIncidentSpectrum(spectra);
    cost->Initialize();

    // Run the optimizer
    typename rtk::ProjectionsDecompositionNegativeLogLikelihood::ParametersType in(this->m_NumberOfMaterials);
    for (unsigned int m = 0; m < this->m_NumberOfMaterials; m++)
      in[m] = inIt.Get()[m];

    // Compute the expected forward projection
    vnl_vector<double> forward = cost->ForwardModel(in);

    // Fill a fixed-sized vector with the computed photon counts in each bin
    typename MeasuredProjectionsType::PixelType outputPixel;
    outputPixel.SetSize(this->m_NumberOfSpectralBins);
    for (unsigned int m = 0; m < this->m_NumberOfSpectralBins; m++)
      outputPixel[m] = forward[m];

    // Write it to the output
    output0It.Set(outputPixel);

    // If requested, store the variances into output(1)
    if (m_ComputeVariances)
    {
      output1It.Set(
        itk::VariableLengthVector<double>(cost->GetVariances(in).data_block(), this->m_NumberOfSpectralBins));
      ++output1It;
    }

    // Move forward
    ++output0It;
    ++inIt;
    ++spectrumIt;
    if (this->GetInputSecondIncidentSpectrum())
      ++secondSpectrumIt;
  }
}

template <typename OutputElementType, typename DetectorResponseImageType, typename ThresholdsType>
vnl_matrix<OutputElementType>
SpectralBinDetectorResponse(const DetectorResponseImageType * drm,
                            const ThresholdsType &            thresholds,
                            const unsigned int                numberOfEnergies)
{
  vnl_matrix<OutputElementType> binnedResponse;
  int                           numberOfSpectralBins = thresholds.GetSize() - 1;
  binnedResponse.set_size(numberOfSpectralBins, numberOfEnergies);
  binnedResponse.fill(0);
  typename DetectorResponseImageType::IndexType indexDet;
  for (unsigned int energy = 0; energy < numberOfEnergies; energy++)
  {
    indexDet[0] = energy;
    for (int bin = 0; bin < numberOfSpectralBins; bin++)
    {
      // First and last couple of values:
      // use trapezoidal rule with linear interpolation
      unsigned int infPulse = itk::Math::floor(thresholds[bin]);
      if (infPulse < 1)
      {
        itkGenericExceptionMacro(<< "Threshold " << thresholds[bin] << " below 0 keV.");
      }
      unsigned int supPulse = itk::Math::floor(thresholds[bin + 1]);
      if (double(supPulse) == thresholds[bin + 1])
        supPulse--;
      if (supPulse - infPulse < 3)
      {
        itkGenericExceptionMacro(<< "Thresholds are too close for the current code.");
      }

      double wInf = infPulse + 1. - thresholds[bin];
      indexDet[1] = infPulse - 1; // Index 0 is 1 keV
      binnedResponse[bin][energy] += 0.5 * wInf * wInf * drm->GetPixel(indexDet);
      indexDet[1]++;
      binnedResponse[bin][energy] += 0.5 * (1. + wInf * (2. - wInf)) * drm->GetPixel(indexDet);

      double wSup = thresholds[bin + 1] - supPulse;
      indexDet[1] = supPulse; // Index 0 is 1 keV
      binnedResponse[bin][energy] += 0.5 * wSup * wSup * drm->GetPixel(indexDet);
      if (supPulse >= drm->GetLargestPossibleRegion().GetSize(1))
      {
        itkGenericExceptionMacro(<< "Threshold " << thresholds[bin + 1] << " above max "
                                 << drm->GetLargestPossibleRegion().GetSize(1) + 1);
      }
      indexDet[1]--;
      binnedResponse[bin][energy] += 0.5 * (1. + wSup * (2. - wSup)) * drm->GetPixel(indexDet);

      // Intermediate values
      for (unsigned int pulseHeight = infPulse + 1; pulseHeight < supPulse - 1; pulseHeight++)
      {
        indexDet[1] = pulseHeight;
        binnedResponse[bin][energy] += drm->GetPixel(indexDet);
      }
    }
  }
  return binnedResponse;
}

} // end namespace rtk

#endif // rtkSpectralForwardModelImageFilter_hxx
