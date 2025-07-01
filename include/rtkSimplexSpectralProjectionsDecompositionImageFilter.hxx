/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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
SimplexSpectralProjectionsDecompositionImageFilter<
  DecomposedProjectionsType,
  MeasuredProjectionsType,
  IncidentSpectrumImageType,
  DetectorResponseImageType,
  MaterialAttenuationsImageType>::SimplexSpectralProjectionsDecompositionImageFilter()
{
  this->SetNumberOfIndexedOutputs(2); // decomposed projections, inverse variance of decomposition noise

  // Decomposed projections
  this->SetNthOutput(0, this->MakeOutput(0));

  // Inverse variance of decomposition noise
  this->SetNthOutput(1, this->MakeOutput(1));

  // Fischer matrix (estimate of inverse covariance of decomposition)
  this->SetNthOutput(2, this->MakeOutput(2));

  // Set the default values of member parameters
  m_NumberOfIterations = 300;
  m_NumberOfMaterials = 4;
  m_NumberOfEnergies = 100;
  m_OptimizeWithRestarts = false;

  // Fill in the vectors and matrices with zeros
  m_MaterialAttenuations.fill(0.); // Not sure this works
  m_DetectorResponse.fill(0.);

  // Initial lengths, set to incorrect values to make sure that they are indeed updated
  m_NumberOfSpectralBins = 8;
  m_OutputInverseCramerRaoLowerBound = false;
  m_OutputFischerMatrix = false;
  m_LogTransformEachBin = false;
  m_GuessInitialization = false;
  m_IsSpectralCT = true;

#ifndef ITK_FUTURE_LEGACY_REMOVE
  // Instantiate the filters required in the overload of SetInputIncidentSpectrum
  m_FlattenFilter = FlattenVectorFilterType::New();
  m_FlattenSecondFilter = FlattenVectorFilterType::New();
  m_PermuteFilter = PermuteFilterType::New();
  m_PermuteSecondFilter = PermuteFilterType::New();
#endif
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::
  SetInputDecomposedProjections(
    const typename itk::ImageBase<DecomposedProjectionsType::ImageDimension> * DecomposedProjections)
{
  // Attempt to dynamic_cast DecomposedProjections into the default DecomposedProjectionsType
  const DecomposedProjectionsType * default_ptr =
    dynamic_cast<const DecomposedProjectionsType *>(DecomposedProjections);
  if (default_ptr)
  {
    this->SetNthInput(0, const_cast<DecomposedProjectionsType *>(default_ptr));
  }
  else
  {
    // Attempt to dynamic_cast DecomposedProjections into one of the supported fixed vector length types
    typedef itk::Image<itk::Vector<DecomposedProjectionsDataType, 1>, DecomposedProjectionsType::ImageDimension> Type1;
    typedef itk::Image<itk::Vector<DecomposedProjectionsDataType, 2>, DecomposedProjectionsType::ImageDimension> Type2;
    typedef itk::Image<itk::Vector<DecomposedProjectionsDataType, 3>, DecomposedProjectionsType::ImageDimension> Type3;
    typedef itk::Image<itk::Vector<DecomposedProjectionsDataType, 4>, DecomposedProjectionsType::ImageDimension> Type4;
    typedef itk::Image<itk::Vector<DecomposedProjectionsDataType, 5>, DecomposedProjectionsType::ImageDimension> Type5;
    const Type1 * ptr1 = dynamic_cast<const Type1 *>(DecomposedProjections);
    const Type2 * ptr2 = dynamic_cast<const Type2 *>(DecomposedProjections);
    const Type3 * ptr3 = dynamic_cast<const Type3 *>(DecomposedProjections);
    const Type4 * ptr4 = dynamic_cast<const Type4 *>(DecomposedProjections);
    const Type5 * ptr5 = dynamic_cast<const Type5 *>(DecomposedProjections);

    if (ptr1)
    {
      this->SetInputFixedVectorLengthDecomposedProjections<1>(ptr1);
    }
    else if (ptr2)
    {
      this->SetInputFixedVectorLengthDecomposedProjections<2>(ptr2);
    }
    else if (ptr3)
    {
      this->SetInputFixedVectorLengthDecomposedProjections<3>(ptr3);
    }
    else if (ptr4)
    {
      this->SetInputFixedVectorLengthDecomposedProjections<4>(ptr4);
    }
    else if (ptr5)
    {
      this->SetInputFixedVectorLengthDecomposedProjections<5>(ptr5);
    }
    else
    {
      itkWarningMacro("The input does not match any of the supported types, and has been ignored");
    }
  }
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
template <unsigned int VNumberOfMaterials>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::
  SetInputFixedVectorLengthDecomposedProjections(
    const itk::Image<itk::Vector<DecomposedProjectionsDataType, VNumberOfMaterials>,
                     DecomposedProjectionsType::ImageDimension> * DecomposedProjections)
{
  using ActualInputType = itk::Image<itk::Vector<DecomposedProjectionsDataType, VNumberOfMaterials>,
                                     DecomposedProjectionsType::ImageDimension>;
  using CastFilterType = itk::CastImageFilter<ActualInputType, DecomposedProjectionsType>;
  auto castPointer = CastFilterType::New();
  castPointer->SetInput(DecomposedProjections);
  castPointer->Update();
  this->SetNthInput(0, const_cast<DecomposedProjectionsType *>(castPointer->GetOutput()));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::
  SetInputMeasuredProjections(
    const typename itk::ImageBase<MeasuredProjectionsType::ImageDimension> * MeasuredProjections)
{
  // Attempt to dynamic_cast MeasuredProjections into the default type MeasuredProjectionsType
  const MeasuredProjectionsType * default_ptr = dynamic_cast<const MeasuredProjectionsType *>(MeasuredProjections);
  if (default_ptr)
  {
    this->SetInput("MeasuredProjections", const_cast<MeasuredProjectionsType *>(default_ptr));
  }
  else
  {
    // Attempt to dynamic_cast MeasuredProjections into one of the supported types
    typedef itk::Image<itk::Vector<MeasuredProjectionsDataType, 1>, MeasuredProjectionsType::ImageDimension> Type1;
    typedef itk::Image<itk::Vector<MeasuredProjectionsDataType, 2>, MeasuredProjectionsType::ImageDimension> Type2;
    typedef itk::Image<itk::Vector<MeasuredProjectionsDataType, 3>, MeasuredProjectionsType::ImageDimension> Type3;
    typedef itk::Image<itk::Vector<MeasuredProjectionsDataType, 4>, MeasuredProjectionsType::ImageDimension> Type4;
    typedef itk::Image<itk::Vector<MeasuredProjectionsDataType, 5>, MeasuredProjectionsType::ImageDimension> Type5;
    typedef itk::Image<itk::Vector<MeasuredProjectionsDataType, 6>, MeasuredProjectionsType::ImageDimension> Type6;
    const Type1 * ptr1 = dynamic_cast<const Type1 *>(MeasuredProjections);
    const Type2 * ptr2 = dynamic_cast<const Type2 *>(MeasuredProjections);
    const Type3 * ptr3 = dynamic_cast<const Type3 *>(MeasuredProjections);
    const Type4 * ptr4 = dynamic_cast<const Type4 *>(MeasuredProjections);
    const Type5 * ptr5 = dynamic_cast<const Type5 *>(MeasuredProjections);
    const Type6 * ptr6 = dynamic_cast<const Type6 *>(MeasuredProjections);

    if (ptr1)
    {
      this->SetInputFixedVectorLengthMeasuredProjections<1>(ptr1);
    }
    else if (ptr2)
    {
      this->SetInputFixedVectorLengthMeasuredProjections<2>(ptr2);
    }
    else if (ptr3)
    {
      this->SetInputFixedVectorLengthMeasuredProjections<3>(ptr3);
    }
    else if (ptr4)
    {
      this->SetInputFixedVectorLengthMeasuredProjections<4>(ptr4);
    }
    else if (ptr5)
    {
      this->SetInputFixedVectorLengthMeasuredProjections<5>(ptr5);
    }
    else if (ptr6)
    {
      this->SetInputFixedVectorLengthMeasuredProjections<6>(ptr6);
    }
    else
    {
      itkWarningMacro("The input does not match any of the supported types, and has been ignored");
    }
  }
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
template <unsigned int VNumberOfSpectralBins>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::
  SetInputFixedVectorLengthMeasuredProjections(
    const itk::Image<itk::Vector<MeasuredProjectionsDataType, VNumberOfSpectralBins>,
                     MeasuredProjectionsType::ImageDimension> * MeasuredProjections)
{
  using ActualInputType = itk::Image<itk::Vector<MeasuredProjectionsDataType, VNumberOfSpectralBins>,
                                     MeasuredProjectionsType::ImageDimension>;
  using CastFilterType = itk::CastImageFilter<ActualInputType, MeasuredProjectionsType>;
  auto castPointer = CastFilterType::New();
  castPointer->SetInput(MeasuredProjections);
  castPointer->UpdateLargestPossibleRegion();
  this->SetInput("MeasuredProjections", const_cast<MeasuredProjectionsType *>(castPointer->GetOutput()));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<
  DecomposedProjectionsType,
  MeasuredProjectionsType,
  IncidentSpectrumImageType,
  DetectorResponseImageType,
  MaterialAttenuationsImageType>::SetDetectorResponse(const DetectorResponseImageType * DetectorResponse)
{
  this->SetInput("DetectorResponse", const_cast<DetectorResponseImageType *>(DetectorResponse));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<
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
void
SimplexSpectralProjectionsDecompositionImageFilter<
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
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::
  SetInputSecondIncidentSpectrum(const IncidentSpectrumImageType * SecondIncidentSpectrum)
{
  this->SetInput("SecondIncidentSpectrum", const_cast<IncidentSpectrumImageType *>(SecondIncidentSpectrum));
}

#ifndef ITK_FUTURE_LEGACY_REMOVE
template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<
  DecomposedProjectionsType,
  MeasuredProjectionsType,
  IncidentSpectrumImageType,
  DetectorResponseImageType,
  MaterialAttenuationsImageType>::SetInputIncidentSpectrum(const VectorSpectrumImageType * IncidentSpectrum)
{
  this->m_FlattenFilter->SetInput(IncidentSpectrum);
  this->m_PermuteFilter->SetInput(this->m_FlattenFilter->GetOutput());
  typename PermuteFilterType::PermuteOrderArrayType order;
  order[0] = 2;
  order[1] = 0;
  order[2] = 1;
  this->m_PermuteFilter->SetOrder(order);
  this->SetInputIncidentSpectrum(m_PermuteFilter->GetOutput());
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<
  DecomposedProjectionsType,
  MeasuredProjectionsType,
  IncidentSpectrumImageType,
  DetectorResponseImageType,
  MaterialAttenuationsImageType>::SetInputSecondIncidentSpectrum(const VectorSpectrumImageType * SecondIncidentSpectrum)
{
  this->m_FlattenSecondFilter->SetInput(SecondIncidentSpectrum);
  this->m_PermuteSecondFilter->SetInput(this->m_FlattenSecondFilter->GetOutput());
  typename PermuteFilterType::PermuteOrderArrayType order;
  order[0] = 2;
  order[1] = 0;
  order[2] = 1;
  this->m_PermuteSecondFilter->SetOrder(order);
  this->SetInputSecondIncidentSpectrum(m_PermuteSecondFilter->GetOutput());
}
#endif

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename DecomposedProjectionsType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::GetInputDecomposedProjections()
{
  return static_cast<const DecomposedProjectionsType *>(this->itk::ProcessObject::GetInput(0));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename MeasuredProjectionsType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::GetInputMeasuredProjections()
{
  return static_cast<const MeasuredProjectionsType *>(this->itk::ProcessObject::GetInput("MeasuredProjections"));
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
typename DetectorResponseImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
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
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
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
typename IncidentSpectrumImageType::ConstPointer
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
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
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
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
itk::DataObject::Pointer
SimplexSpectralProjectionsDecompositionImageFilter<
  DecomposedProjectionsType,
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
    case 2:
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
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();
  this->GetOutput(0)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());
  this->GetOutput(1)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());
  this->GetOutput(2)->SetLargestPossibleRegion(this->GetInputDecomposedProjections()->GetLargestPossibleRegion());

  this->m_NumberOfSpectralBins = this->GetInputMeasuredProjections()->GetVectorLength();
  this->m_NumberOfMaterials = this->GetInputDecomposedProjections()->GetVectorLength();
  this->m_NumberOfEnergies = this->GetInputIncidentSpectrum()->GetLargestPossibleRegion().GetSize(0);

  // Set vector length for the fischer matrix
  this->GetOutput(2)->SetVectorLength(this->m_NumberOfMaterials * this->m_NumberOfMaterials);

  // Change vector length for the decomposed projections, if required
  if (m_LogTransformEachBin)
    this->GetOutput(0)->SetVectorLength(this->m_NumberOfMaterials + this->m_NumberOfSpectralBins);
}

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();

  // Input 0 is the initial decomposed projections
  typename DecomposedProjectionsType::Pointer inputPtr0 =
    const_cast<DecomposedProjectionsType *>(this->GetInputDecomposedProjections().GetPointer());
  if (!inputPtr0)
    return;
  inputPtr0->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

  // Input 1 is the spectral projections
  typename MeasuredProjectionsType::Pointer inputPtr1 =
    const_cast<MeasuredProjectionsType *>(this->GetInputMeasuredProjections().GetPointer());
  if (!inputPtr1)
    return;
  inputPtr1->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

  // Input 2 is the incident spectrum image
  typename IncidentSpectrumImageType::Pointer inputPtr2 =
    const_cast<IncidentSpectrumImageType *>(this->GetInputIncidentSpectrum().GetPointer());
  if (!inputPtr2)
    return;

  typename IncidentSpectrumImageType::RegionType requested =
    this->GetInputIncidentSpectrum()->GetLargestPossibleRegion();
  typename IncidentSpectrumImageType::IndexType indexRequested = requested.GetIndex();
  typename IncidentSpectrumImageType::SizeType  sizeRequested = requested.GetSize();
  for (unsigned int i = 0; i < IncidentSpectrumImageType::GetImageDimension() - 1; i++)
  {
    indexRequested[i + 1] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    sizeRequested[i + 1] = this->GetOutput()->GetRequestedRegion().GetSize()[i];
  }

  inputPtr2->SetRequestedRegion(requested);

  // Input 3 is the detector response image (2D float)
  typename DetectorResponseImageType::Pointer inputPtr3 =
    const_cast<DetectorResponseImageType *>(this->GetDetectorResponse().GetPointer());
  if (!inputPtr3)
    return;
  inputPtr3->SetRequestedRegion(inputPtr3->GetLargestPossibleRegion());

  // Input 4 is the material attenuations image (2D float)
  typename MaterialAttenuationsImageType::Pointer inputPtr4 =
    const_cast<MaterialAttenuationsImageType *>(this->GetMaterialAttenuations().GetPointer());
  if (!inputPtr4)
    return;
  inputPtr4->SetRequestedRegion(inputPtr4->GetLargestPossibleRegion());
}


template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType,
          typename DetectorResponseImageType,
          typename MaterialAttenuationsImageType>
void
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  // Read the material attenuations image as a matrix
  typename MaterialAttenuationsImageType::IndexType indexMat;
  this->m_MaterialAttenuations.set_size(this->m_NumberOfEnergies, this->m_NumberOfMaterials);
  for (unsigned int energy = 0; energy < this->m_NumberOfEnergies; energy++)
  {
    indexMat[1] = energy;
    for (unsigned int material = 0; material < this->m_NumberOfMaterials; material++)
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
SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                   MeasuredProjectionsType,
                                                   IncidentSpectrumImageType,
                                                   DetectorResponseImageType,
                                                   MaterialAttenuationsImageType>::
  DynamicThreadedGenerateData(const typename DecomposedProjectionsType::RegionType & outputRegionForThread)
{
  ////////////////////////////////////////////////////////////////////
  // Create a Nelder-Mead simplex optimizer and its cost function
  auto                                                        optimizer = itk::AmoebaOptimizer::New();
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
  if (m_GuessInitialization && (!this->GetInputSecondIncidentSpectrum()))
    cost->SetThresholds(this->m_Thresholds);

  // Pass the binned detector response to the cost function
  cost->SetDetectorResponse(this->m_DetectorResponse);

  // Set the optimizer
  optimizer->SetCostFunction(cost);
  optimizer->SetMaximumNumberOfIterations(this->m_NumberOfIterations);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Walk the output projection stack. For each pixel, set the cost function's member variables and run the optimizer.
  itk::ImageRegionIterator<DecomposedProjectionsType>      output0It(this->GetOutput(0), outputRegionForThread);
  itk::ImageRegionIterator<DecomposedProjectionsType>      output1It(this->GetOutput(1), outputRegionForThread);
  itk::ImageRegionIterator<DecomposedProjectionsType>      output2It(this->GetOutput(2), outputRegionForThread);
  itk::ImageRegionConstIterator<DecomposedProjectionsType> inputIt(this->GetInputDecomposedProjections(),
                                                                   outputRegionForThread);
  itk::ImageRegionConstIterator<MeasuredProjectionsType>   spectralProjIt(this->GetInputMeasuredProjections(),
                                                                        outputRegionForThread);

  typename IncidentSpectrumImageType::RegionType incidentSpectrumRegionForThread =
    this->GetInputIncidentSpectrum()->GetLargestPossibleRegion();
  for (unsigned int dim = 0; dim < IncidentSpectrumImageType::GetImageDimension() - 1; dim++)
  {
    incidentSpectrumRegionForThread.SetIndex(dim + 1, outputRegionForThread.GetIndex()[dim]);
    incidentSpectrumRegionForThread.SetSize(dim + 1, outputRegionForThread.GetSize()[dim]);
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
    // The input incident spectrum image typically has lower dimension than
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
      for (unsigned int e = 0; e < m_NumberOfEnergies; e++)
      {
        spectra.put(0, e, spectrumIt.Get());
        spectra.put(1, e, secondSpectrumIt.Get());
        ++spectrumIt;
        ++secondSpectrumIt;
      }
    }
    else
    {
      spectra.set_size(1, m_NumberOfEnergies);
      for (unsigned int e = 0; e < m_NumberOfEnergies; e++)
      {
        spectra.put(0, e, spectrumIt.Get());
        ++spectrumIt;
      }
    }

    // Pass the incident spectrum vector to cost function
    cost->SetIncidentSpectrum(spectra);
    cost->Initialize();

    // Pass the detector counts vector to cost function
    cost->SetMeasuredData(spectralProjIt.Get());

    // Run the optimizer
    typename rtk::ProjectionsDecompositionNegativeLogLikelihood::ParametersType startingPosition(
      this->m_NumberOfMaterials);
    if (m_GuessInitialization)
    {
      itk::VariableLengthVector<double> guess = cost->GuessInitialization();
      for (unsigned int m = 0; m < this->m_NumberOfMaterials; m++)
        startingPosition[m] = guess[m];
    }
    else
    {
      for (unsigned int m = 0; m < this->m_NumberOfMaterials; m++)
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
      for (unsigned int bin = 0; bin < this->m_NumberOfSpectralBins; bin++)
        outputPixel[bin + this->m_NumberOfMaterials] = cost->BinwiseLogTransform()[bin];
    }
    else
      outputPixel.SetSize(this->m_NumberOfMaterials);

    for (unsigned int m = 0; m < this->m_NumberOfMaterials; m++)
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
  }
}

} // end namespace rtk

#endif // rtkSimplexSpectralProjectionsDecompositionImageFilter_hxx
