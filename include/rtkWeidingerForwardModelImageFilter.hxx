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
#ifndef rtkWeidingerForwardModelImageFilter_hxx
#define rtkWeidingerForwardModelImageFilter_hxx

#include "rtkWeidingerForwardModelImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

namespace rtk
{
//
// Constructor
//
template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::
  WeidingerForwardModelImageFilter()
{
  this->SetNumberOfRequiredInputs(4);

  this->SetNthOutput(0, this->MakeOutput(0));
  this->SetNthOutput(1, this->MakeOutput(1));
}


template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
void
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::
  SetInputMaterialProjections(const TMaterialProjections * materialProjections)
{
  this->SetNthInput(0, const_cast<TMaterialProjections *>(materialProjections));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
void
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::SetInputPhotonCounts(
  const TPhotonCounts * photonCounts)
{
  this->SetNthInput(1, const_cast<TPhotonCounts *>(photonCounts));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
void
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::SetInputSpectrum(
  const TSpectrum * spectrum)
{
  this->SetNthInput(2, const_cast<TSpectrum *>(spectrum));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
void
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::
  SetInputProjectionsOfOnes(const TProjections * projectionsOfOnes)
{
  this->SetNthInput(3, const_cast<TProjections *>(projectionsOfOnes));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
typename TMaterialProjections::ConstPointer
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::
  GetInputMaterialProjections()
{
  return static_cast<const TMaterialProjections *>(this->itk::ProcessObject::GetInput(0));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
typename TPhotonCounts::ConstPointer
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::GetInputPhotonCounts()
{
  return static_cast<const TPhotonCounts *>(this->itk::ProcessObject::GetInput(1));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
typename TSpectrum::ConstPointer
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::GetInputSpectrum()
{
  return static_cast<const TSpectrum *>(this->itk::ProcessObject::GetInput(2));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
typename TProjections::ConstPointer
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::
  GetInputProjectionsOfOnes()
{
  return static_cast<const TProjections *>(this->itk::ProcessObject::GetInput(3));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
itk::ProcessObject::DataObjectPointer
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::MakeOutput(
  itk::ProcessObject::DataObjectPointerArraySizeType idx)
{
  itk::DataObject::Pointer output;

  switch (idx)
  {
    case 0:
      output = (TOutputImage1::New()).GetPointer();
      break;
    case 1:
      output = (TOutputImage2::New()).GetPointer();
      break;
    default:
      std::cerr << "No output " << idx << std::endl;
      output = nullptr;
      break;
  }
  return output.GetPointer();
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
itk::ProcessObject::DataObjectPointer
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::MakeOutput(
  const itk::ProcessObject::DataObjectIdentifierType & idx)
{
  return Superclass::MakeOutput(idx);
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
typename WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::TOutputImage1 *
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::GetOutput1()
{
  return dynamic_cast<TOutputImage1 *>(this->itk::ProcessObject::GetOutput(0));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
typename WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::TOutputImage2 *
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::GetOutput2()
{
  return dynamic_cast<TOutputImage2 *>(this->itk::ProcessObject::GetOutput(1));
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
void
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::
  SetBinnedDetectorResponse(const BinnedDetectorResponseType & detResp)
{
  bool modified = false;

  unsigned int nEnergies = detResp.columns();
  if (m_BinnedDetectorResponse.columns() != nEnergies)
  {
    modified = true;
    m_BinnedDetectorResponse.set_size(nBins, nEnergies);
    m_BinnedDetectorResponse.fill(0.);
  }

  for (unsigned int r = 0; r < nBins; r++)
  {
    for (unsigned int c = 0; c < nEnergies; c++)
    {
      if (itk::Math::NotExactlyEquals(m_BinnedDetectorResponse[r][c], detResp[r][c]))
      {
        m_BinnedDetectorResponse[r][c] = detResp[r][c];
        modified = true;
      }
    }
  }
  if (modified)
  {
    this->Modified();
  }
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
void
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::SetMaterialAttenuations(
  const MaterialAttenuationsType & matAtt)
{
  bool modified = false;

  unsigned int nEnergies = matAtt.rows();
  if (m_MaterialAttenuations.rows() != nEnergies)
  {
    modified = true;
    m_MaterialAttenuations.set_size(nEnergies, nMaterials);
    m_MaterialAttenuations.fill(0.);
  }

  for (unsigned int r = 0; r < nEnergies; r++)
  {
    for (unsigned int c = 0; c < nMaterials; c++)
    {
      if (itk::Math::NotExactlyEquals(m_MaterialAttenuations[r][c], matAtt[r][c]))
      {
        m_MaterialAttenuations[r][c] = matAtt[r][c];
        modified = true;
      }
    }
  }
  if (modified)
  {
    this->Modified();
  }
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
void
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::
  GenerateInputRequestedRegion()
{
  // Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Get the requested regions on both outputs (should be identical)
  typename TOutputImage1::RegionType outputRequested1 = this->GetOutput1()->GetRequestedRegion();
  typename TOutputImage2::RegionType outputRequested2 = this->GetOutput2()->GetRequestedRegion();
  if (outputRequested1 != outputRequested2)
    itkGenericExceptionMacro(
      << "In rtkWeidingerForwardModelImageFilter: requested regions for outputs 1 and 2 should be identical");

  // Get pointers to the inputs
  typename TMaterialProjections::Pointer input1Ptr =
    const_cast<TMaterialProjections *>(this->GetInputMaterialProjections().GetPointer());
  typename TPhotonCounts::Pointer input2Ptr = const_cast<TPhotonCounts *>(this->GetInputPhotonCounts().GetPointer());
  typename TSpectrum::Pointer     input3Ptr = const_cast<TSpectrum *>(this->GetInputSpectrum().GetPointer());
  typename TProjections::Pointer input4Ptr = const_cast<TProjections *>(this->GetInputProjectionsOfOnes().GetPointer());

  // The first and second input must have the same requested region as the outputs
  input1Ptr->SetRequestedRegion(outputRequested1);
  input2Ptr->SetRequestedRegion(outputRequested1);
  input4Ptr->SetRequestedRegion(outputRequested1);

  // The spectrum input's first dimension is the energy,
  // and then come the spatial dimensions. The projection
  // number is discarded
  // Compute the requested region for the spectrum
  typename TSpectrum::RegionType spectrumRegion = input3Ptr->GetLargestPossibleRegion();
  for (unsigned int d = 0; d < TSpectrum::ImageDimension - 1; d++)
  {
    spectrumRegion.SetIndex(d + 1, outputRequested1.GetIndex()[d]);
    spectrumRegion.SetSize(d + 1, outputRequested1.GetSize()[d]);
  }

  // Set the requested region for the spectrum
  input3Ptr->SetRequestedRegion(spectrumRegion);
}

template <class TMaterialProjections, class TPhotonCounts, class TSpectrum, class TProjections>
void
WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>::
  DynamicThreadedGenerateData(const typename TOutputImage1::RegionType & outputRegionForThread)
{
  // Create the region corresponding to outputRegionForThread for the spectrum input
  typename TSpectrum::RegionType spectrumRegion = this->GetInputSpectrum()->GetLargestPossibleRegion();
  for (unsigned int d = 0; d < TSpectrum::ImageDimension - 1; d++)
  {
    spectrumRegion.SetIndex(d + 1, outputRegionForThread.GetIndex()[d]);
    spectrumRegion.SetSize(d + 1, outputRegionForThread.GetSize()[d]);
  }
  unsigned int nEnergies = spectrumRegion.GetSize()[0];

  // Create iterators for all inputs and outputs
  itk::ImageRegionIterator<TOutputImage1>             out1It(this->GetOutput1(), outputRegionForThread);
  itk::ImageRegionIterator<TOutputImage2>             out2It(this->GetOutput2(), outputRegionForThread);
  itk::ImageRegionConstIterator<TMaterialProjections> projIt(this->GetInputMaterialProjections(),
                                                             outputRegionForThread);
  itk::ImageRegionConstIterator<TPhotonCounts> photonCountsIt(this->GetInputPhotonCounts(), outputRegionForThread);
  itk::ImageRegionConstIterator<TSpectrum>     spectrumIt(this->GetInputSpectrum(), spectrumRegion);
  itk::ImageRegionConstIterator<TProjections>  projOfOnesIt(this->GetInputProjectionsOfOnes(), outputRegionForThread);

  // Declare intermediate variables
  vnl_vector<dataType>                           spectrum(nEnergies);
  vnl_matrix<dataType>                           efficientSpectrum(nBins, nEnergies);
  vnl_vector<dataType>                           attenuationFactors(nEnergies);
  vnl_vector<dataType>                           expectedCounts(nBins);
  vnl_vector<dataType>                           oneMinusRatios(nBins);
  vnl_matrix<dataType>                           intermForGradient(nEnergies, nMaterials);
  vnl_matrix<dataType>                           interm2ForGradient(nBins, nMaterials);
  itk::Vector<dataType, nMaterials>              forOutput1;
  vnl_matrix<dataType>                           intermForHessian(nEnergies, nMaterials * nMaterials);
  vnl_matrix<dataType>                           interm2ForHessian(nBins, nMaterials * nMaterials);
  itk::Vector<dataType, nMaterials * nMaterials> forOutput2;

  while (!out1It.IsAtEnd())
  {
    // After each projection, the spectrum's iterator must come back to the beginning
    if (spectrumIt.IsAtEnd())
      spectrumIt.GoToBegin();

    // Read the spectrum at the current pixel
    for (unsigned int e = 0; e < nEnergies; e++)
    {
      spectrum[e] = spectrumIt.Get();
      ++spectrumIt;
    }

    // Get efficient spectrum, by equivalent of element-wise product with implicit extension
    for (unsigned int r = 0; r < nBins; r++)
      for (unsigned int c = 0; c < nEnergies; c++)
        efficientSpectrum[r][c] = m_BinnedDetectorResponse[r][c] * spectrum[c];

    // Get attenuation factors at each energy from material projections
    attenuationFactors = m_MaterialAttenuations * projIt.Get().GetVnlVector();
    for (unsigned int r = 0; r < nEnergies; r++)
      attenuationFactors[r] = std::exp(-attenuationFactors[r]);

    // Get the expected photon counts through these attenuations
    expectedCounts = efficientSpectrum * attenuationFactors;

    // Get intermediate variables used in the computation of the first output
    for (unsigned int r = 0; r < nBins; r++)
      oneMinusRatios[r] = 1 - (photonCountsIt.Get()[r] / expectedCounts[r]);

    // Form an intermediate variable used for the gradient of the cost function,
    // (the derivation of the exponential implies that a m_MaterialAttenuations
    // gets out), by equivalent of element-wise product with implicit extension
    for (unsigned int r = 0; r < nEnergies; r++)
      for (unsigned int c = 0; c < nMaterials; c++)
        intermForGradient[r][c] = m_MaterialAttenuations[r][c] * attenuationFactors[r];

    // Multiply by the spectrum
    interm2ForGradient = -efficientSpectrum * intermForGradient;

    // Compute the product with oneMinusRatios, with implicit extension
    for (unsigned int r = 0; r < nBins; r++)
      for (unsigned int c = 0; c < nMaterials; c++)
        interm2ForGradient[r][c] *= oneMinusRatios[r];

    // Finally, compute the vector to be written in first output
    // by summing on the bins
    forOutput1.Fill(0);
    for (unsigned int r = 0; r < nBins; r++)
      for (unsigned int c = 0; c < nMaterials; c++)
        forOutput1[c] += interm2ForGradient[r][c];

    // Set the output1
    out1It.Set(forOutput1);

    // Now compute output2

    // Form an intermediate variable used for the hessian of the cost function,
    // (the double derivation of the exponential implies that a m_MaterialAttenuations^2
    // gets out), by equivalent of element-wise product with implicit extension
    for (unsigned int r = 0; r < nEnergies; r++)
      for (unsigned int c = 0; c < nMaterials; c++)
        for (unsigned int c2 = 0; c2 < nMaterials; c2++)
          intermForHessian[r][c2 + nMaterials * c] =
            m_MaterialAttenuations[r][c] * m_MaterialAttenuations[r][c2] * attenuationFactors[r];

    // Multiply by the spectrum
    interm2ForHessian = efficientSpectrum * intermForHessian;

    // Sum on the bins
    forOutput2.Fill(0);
    for (unsigned int r = 0; r < nBins; r++)
      for (unsigned int c = 0; c < nMaterials * nMaterials; c++)
        forOutput2[c] += interm2ForHessian[r][c];

    // Multiply by the projection of ones
    forOutput2 *= projOfOnesIt.Get();

    // Set the output2
    out2It.Set(forOutput2);

    ++out1It;
    ++out2It;
    ++projIt;
    ++photonCountsIt;
    ++projOfOnesIt;
  }
}

} // namespace rtk

#endif
