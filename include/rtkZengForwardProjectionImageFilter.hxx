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

#ifndef rtkZengForwardProjectionImageFilter_hxx
#define rtkZengForwardProjectionImageFilter_hxx

#include <math.h>

#include "rtkZengForwardProjectionImageFilter.h"

#include "rtkHomogeneousMatrix.h"
#include "rtkBoxShape.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"
#include "itkImageFileWriter.h"

#include <itkImageRegionIteratorWithIndex.h>
#include <itkInputDataObjectConstIterator.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
ZengForwardProjectionImageFilter<TInputImage, TOutputImage>::ZengForwardProjectionImageFilter()
{
  // Set default parameters
  m_SigmaZero = 1.5417233052142099;
  m_Alpha = 0.016241189545787734;
  m_VectorOrthogonalDetector[0] = 0;
  m_VectorOrthogonalDetector[1] = 0;
  m_VectorOrthogonalDetector[2] = 1;
  m_centerVolume.Fill(0);

  // Create each filter of the composite filter
  m_RegionOfInterest = RegionOfInterestFilterType::New();
  m_AddImageFilter = AddImageFilterType::New();
  m_PasteImageFilter = PasteImageFilterType::New();
  m_DiscreteGaussianFilter = DiscreteGaussianFilterType::New();
  m_ResampleImageFilter = ResampleImageFilterType::New();
  m_Transform = TransformType::New();
  m_ChangeInformation = ChangeInformationFilterType::New();
  m_MultiplyImageFilter = MultiplyImageFilterType::New();
  m_AttenuationMapExpImageFilter = nullptr;
  m_AttenuationMapMultiplyImageFilter = nullptr;
  m_AttenuationMapRegionOfInterest = nullptr;
  m_AttenuationMapResampleImageFilter = nullptr;
  m_AttenuationMapConstantMultiplyImageFilter = nullptr;
  m_AttenuationMapChangeInformation = nullptr;

  // Permanent internal connections
  m_AddImageFilter->SetInput1(m_DiscreteGaussianFilter->GetOutput());
  m_AddImageFilter->SetInput2(m_ChangeInformation->GetOutput());
  m_MultiplyImageFilter->SetInput(m_PasteImageFilter->GetOutput());

  // Default parameters
  m_DiscreteGaussianFilter->SetMaximumError(0.00001);
  m_DiscreteGaussianFilter->SetFilterDimensionality(2);
#if (ITK_VERSION_MAJOR == 5) && (ITK_VERSION_MINOR >= 1)
  m_DiscreteGaussianFilter->SetInputBoundaryCondition(&m_BoundsCondition);
  m_DiscreteGaussianFilter->SetRealBoundaryCondition(&m_BoundsCondition);
#endif
  m_ChangeInformation->ChangeOriginOn();
  m_ChangeInformation->SetReferenceImage(m_DiscreteGaussianFilter->GetOutput());
  m_ChangeInformation->SetUseReferenceImage(true);
}

template <class TInputImage, class TOutputImage>
void
ZengForwardProjectionImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();
  // Input 2 is the attenuation map relative to the volume
  typename Superclass::InputImagePointer inputPtr2 = const_cast<TInputImage *>(this->GetInput(2));
  if (!inputPtr2)
    return;
}

template <class TInputImage, class TOutputImage>
void
ZengForwardProjectionImageFilter<TInputImage, TOutputImage>::VerifyInputInformation() const
{
  using ImageBaseType = const itk::ImageBase<InputImageDimension>;

  ImageBaseType * inputPtr1 = nullptr;

  itk::InputDataObjectConstIterator it(this);
  for (; !it.IsAtEnd(); ++it)
  {
    // Check whether the output is an image of the appropriate
    // dimension (use ProcessObject's version of the GetInput()
    // method since it returns the input as a pointer to a
    // DataObject as opposed to the subclass version which
    // static_casts the input to an TInputImage).
    if (it.GetName() != "Primary")
    {
      inputPtr1 = dynamic_cast<ImageBaseType *>(it.GetInput());
    }
    if (inputPtr1)
    {
      break;
    }
  }

  for (; !it.IsAtEnd(); ++it)
  {
    if (it.GetName() != "Primary")
    {
      auto * inputPtrN = dynamic_cast<ImageBaseType *>(it.GetInput());
      // Physical space computation only matters if we're using two
      // images, and not an image and a constant.
      if (inputPtrN)
      {
        // check that the image occupy the same physical space, and that
        // each index is at the same physical location

        // tolerance for origin and spacing depends on the size of pixel
        // tolerance for directions a fraction of the unit cube.
        const double coordinateTol = itk::Math::abs(Self::GetGlobalDefaultCoordinateTolerance() *
                                                    inputPtr1->GetSpacing()[0]); // use first dimension spacing

        if (!inputPtr1->GetOrigin().GetVnlVector().is_equal(inputPtrN->GetOrigin().GetVnlVector(), coordinateTol) ||
            !inputPtr1->GetSpacing().GetVnlVector().is_equal(inputPtrN->GetSpacing().GetVnlVector(), coordinateTol) ||
            !inputPtr1->GetDirection().GetVnlMatrix().as_ref().is_equal(
              inputPtrN->GetDirection().GetVnlMatrix().as_ref(), Self::GetGlobalDefaultDirectionTolerance()))
        {
          std::ostringstream originString, spacingString, directionString;
          if (!inputPtr1->GetOrigin().GetVnlVector().is_equal(inputPtrN->GetOrigin().GetVnlVector(), coordinateTol))
          {
            originString.setf(std::ios::scientific);
            originString.precision(7);
            originString << "InputImage Origin: " << inputPtr1->GetOrigin() << ", InputImage" << it.GetName()
                         << " Origin: " << inputPtrN->GetOrigin() << std::endl;
            originString << "\tTolerance: " << coordinateTol << std::endl;
          }
          if (!inputPtr1->GetSpacing().GetVnlVector().is_equal(inputPtrN->GetSpacing().GetVnlVector(), coordinateTol))
          {
            spacingString.setf(std::ios::scientific);
            spacingString.precision(7);
            spacingString << "InputImage Spacing: " << inputPtr1->GetSpacing() << ", InputImage" << it.GetName()
                          << " Spacing: " << inputPtrN->GetSpacing() << std::endl;
            spacingString << "\tTolerance: " << coordinateTol << std::endl;
          }
          if (!inputPtr1->GetDirection().GetVnlMatrix().as_ref().is_equal(
                inputPtrN->GetDirection().GetVnlMatrix().as_ref(), Self::GetGlobalDefaultDirectionTolerance()))
          {
            directionString.setf(std::ios::scientific);
            directionString.precision(7);
            directionString << "InputImage Direction: " << inputPtr1->GetDirection() << ", InputImage" << it.GetName()
                            << " Direction: " << inputPtrN->GetDirection() << std::endl;
            directionString << "\tTolerance: " << Self::GetGlobalDefaultDirectionTolerance() << std::endl;
          }
          itkExceptionMacro(<< "Inputs do not occupy the same physical space! " << std::endl
                            << originString.str() << spacingString.str() << directionString.str());
        }
      }
    }
  }
}

template <class TInputImage, class TOutputImage>
void
ZengForwardProjectionImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{

  // Info of the input volume
  const typename OuputCPUImageType::PointType   originVolume = this->GetInput(1)->GetOrigin();
  const typename OuputCPUImageType::SpacingType spacingVolume = this->GetInput(1)->GetSpacing();
  const typename OuputCPUImageType::SizeType    sizeVolume = this->GetInput(1)->GetLargestPossibleRegion().GetSize();

  // Info of the input stack of projections
  const typename InputCPUImageType::SpacingType spacingProjections = this->GetInput(0)->GetSpacing();
  const typename InputCPUImageType::SizeType  sizeProjection = this->GetInput(0)->GetLargestPossibleRegion().GetSize();
  const typename OuputCPUImageType::PointType originProjection = this->GetInput(0)->GetOrigin();

  // Find the center of the volume
  m_centerVolume[0] = originVolume[0] + spacingVolume[0] * (double)(sizeVolume[0] - 1) / 2.0;
  m_centerVolume[1] = originVolume[1] + spacingVolume[1] * (double)(sizeVolume[1] - 1) / 2.0;
  m_centerVolume[2] = originVolume[2] + spacingVolume[2] * (double)(sizeVolume[2] - 1) / 2.0;

  PointType centerRotation;
  centerRotation.Fill(0);
  m_Transform->SetCenter(centerRotation);
  m_ResampleImageFilter->SetTransform(m_Transform);
  m_ResampleImageFilter->SetOutputParametersFromImage(this->GetInput(1));

  // Set the output size of the rotated volume
  typename OuputCPUImageType::SizeType outputSize;
  outputSize[0] = sizeProjection[0];
  outputSize[1] = sizeProjection[1];
  outputSize[2] = sizeVolume[2] * std::sqrt(2);
  m_ResampleImageFilter->SetSize(outputSize);

  // Set the new origin of the rotated volume
  typename OuputCPUImageType::PointType outputOrigin;
  outputOrigin[0] = originProjection[0];
  outputOrigin[1] = originProjection[1];
  outputOrigin[2] = m_centerVolume[2] - spacingVolume[2] * (double)(outputSize[2] - 1) / 2.0;
  m_ResampleImageFilter->SetOutputOrigin(outputOrigin);

  // Set the output spacing of the rotated volume
  typename OuputCPUImageType::SpacingType outputSpacing;
  outputSpacing[0] = spacingProjections[0];
  outputSpacing[1] = spacingProjections[1];
  outputSpacing[2] = spacingVolume[2];
  m_ResampleImageFilter->SetOutputSpacing(outputSpacing);
  m_ResampleImageFilter->SetInput(this->GetInput(1));
  m_ResampleImageFilter->UpdateOutputInformation();

  // We only set the first sub-stack at that point, the rest will be
  // requested in the GenerateData function
  typename RegionOfInterestFilterType::InputImageRegionType projRegion;
  const unsigned int                                        Dimension = this->InputImageDimension;
  projRegion = m_ResampleImageFilter->GetOutput()->GetLargestPossibleRegion();
  projRegion.SetSize(Dimension - 1, 1);
  m_RegionOfInterest->SetRegionOfInterest(projRegion);
  m_RegionOfInterest->SetInput(m_ResampleImageFilter->GetOutput());
  m_RegionOfInterest->UpdateOutputInformation();

  m_DiscreteGaussianFilter->SetVariance(pow(m_SigmaZero, 2.0));

  if (!(this->GetInput(2)))
  {
    m_DiscreteGaussianFilter->SetInput(m_RegionOfInterest->GetOutput());
  }
  else
  {
    m_AttenuationMapExpImageFilter = ExpImageFilterType::New();
    m_AttenuationMapMultiplyImageFilter = MultiplyImageFilterType::New();
    m_AttenuationMapRegionOfInterest = RegionOfInterestFilterType::New();
    m_AttenuationMapResampleImageFilter = ResampleImageFilterType::New();
    m_AttenuationMapConstantMultiplyImageFilter = MultiplyImageFilterType::New();
    m_AttenuationMapChangeInformation = ChangeInformationFilterType::New();
    m_AttenuationMapChangeInformation->ChangeOriginOn();
    m_AttenuationMapChangeInformation->SetReferenceImage(m_DiscreteGaussianFilter->GetOutput());
    m_AttenuationMapChangeInformation->SetUseReferenceImage(true);

    m_AttenuationMapResampleImageFilter->SetTransform(m_Transform);
    m_AttenuationMapResampleImageFilter->SetOutputParametersFromImage(this->GetInput(1));
    m_AttenuationMapResampleImageFilter->SetSize(outputSize);
    m_AttenuationMapResampleImageFilter->SetOutputOrigin(outputOrigin);
    m_AttenuationMapResampleImageFilter->SetOutputSpacing(outputSpacing);
    m_AttenuationMapResampleImageFilter->SetInput(this->GetInput(2));
    m_AttenuationMapResampleImageFilter->UpdateOutputInformation();

    m_AttenuationMapRegionOfInterest->SetRegionOfInterest(projRegion);
    m_AttenuationMapRegionOfInterest->SetInput(m_AttenuationMapResampleImageFilter->GetOutput());
    m_AttenuationMapRegionOfInterest->UpdateOutputInformation();

    m_AttenuationMapConstantMultiplyImageFilter->SetInput(m_AttenuationMapRegionOfInterest->GetOutput());
    m_AttenuationMapConstantMultiplyImageFilter->SetConstant(-spacingVolume[2]);
    m_AttenuationMapExpImageFilter->SetInput(m_AttenuationMapConstantMultiplyImageFilter->GetOutput());
    m_AttenuationMapMultiplyImageFilter->SetInput1(m_RegionOfInterest->GetOutput());
    m_AttenuationMapMultiplyImageFilter->SetInput2(m_AttenuationMapExpImageFilter->GetOutput());
    m_DiscreteGaussianFilter->SetInput(m_AttenuationMapMultiplyImageFilter->GetOutput());
  }

  m_MultiplyImageFilter->SetConstant(spacingVolume[2]);

  m_PasteImageFilter->SetSourceImage(m_DiscreteGaussianFilter->GetOutput());
  m_PasteImageFilter->SetDestinationImage(this->GetInput(0));
  m_PasteImageFilter->SetSourceRegion(m_DiscreteGaussianFilter->GetOutput()->GetLargestPossibleRegion());

  // Update output information
  m_PasteImageFilter->UpdateOutputInformation();
  this->GetOutput()->SetOrigin(m_PasteImageFilter->GetOutput()->GetOrigin());
  this->GetOutput()->SetSpacing(m_PasteImageFilter->GetOutput()->GetSpacing());
  this->GetOutput()->SetDirection(m_PasteImageFilter->GetOutput()->GetDirection());
  this->GetOutput()->SetLargestPossibleRegion(m_PasteImageFilter->GetOutput()->GetLargestPossibleRegion());

  m_ResampleImageFilter->ReleaseDataFlagOn();
}

template <class TInputImage, class TOutputImage>
void
ZengForwardProjectionImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  const unsigned int                                    Dimension = this->InputImageDimension;
  const typename Superclass::GeometryType::ConstPointer geometry = this->GetGeometry();
  typename OuputCPUImageType::RegionType                projRegion = this->GetInput(0)->GetLargestPossibleRegion();
  std::vector<double>                                   list_angle;
  if (geometry->GetGantryAngles().size() != projRegion.GetSize(Dimension - 1))
  {
    list_angle.push_back(geometry->GetGantryAngles()[projRegion.GetIndex(Dimension - 1)]);
  }
  else
  {
    list_angle = geometry->GetGantryAngles();
  }

  typename OuputCPUImageType::Pointer   currentSlice;
  typename OuputCPUImageType::Pointer   pimg;
  typename OuputCPUImageType::Pointer   rotatedVolume;
  typename OuputCPUImageType::IndexType indexSlice;
  typename OuputCPUImageType::IndexType indexProjection;
  PointType                             pointSlice;
  PointType                             centerRotatedVolume;
  PointType                             originRotatedVolume;

  indexSlice.Fill(0);
  indexProjection.Fill(0);
  float dist, sigmaSlice, thicknessSlice;
  int   nbProjections = 0;
  for (auto & angle : list_angle)
  {
    // Set the rotation angle.
    m_Transform->SetRotation(0., angle, 0.);
    centerRotatedVolume = m_Transform->GetMatrix() * m_centerVolume;

    // Rotate the input volume
    this->GetInput(1)->GetBufferPointer();
    originRotatedVolume = m_ResampleImageFilter->GetOutputOrigin();
    originRotatedVolume[2] = centerRotatedVolume[2] - m_ResampleImageFilter->GetOutputSpacing()[2] *
                                                        (double)(m_ResampleImageFilter->GetSize()[2] - 1) / 2.0;
    m_ResampleImageFilter->SetOutputOrigin(originRotatedVolume);
    m_ResampleImageFilter->Update();
    rotatedVolume = m_ResampleImageFilter->GetOutput();
    rotatedVolume->DisconnectPipeline();
    thicknessSlice = rotatedVolume->GetSpacing()[2];

    // Extract slice from the volume starting by the farthest from the detector.
    typename RegionOfInterestFilterType::InputImageRegionType desiredRegion = rotatedVolume->GetLargestPossibleRegion();
    unsigned int                                              nbSlice = desiredRegion.GetSize(Dimension - 1);
    desiredRegion.SetSize(Dimension - 1, 1);
    desiredRegion.SetIndex(Dimension - 1, 0);
    m_RegionOfInterest->SetInput(rotatedVolume);
    m_RegionOfInterest->SetRegionOfInterest(desiredRegion);
    m_RegionOfInterest->UpdateOutputInformation();
    m_RegionOfInterest->Update();
    if (!(this->GetInput(2)))
    {
      currentSlice = m_RegionOfInterest->GetOutput();
    }
    else
    {
      typename OuputCPUImageType::Pointer rotatedAttenuation;
      m_AttenuationMapResampleImageFilter->SetOutputOrigin(originRotatedVolume);
      m_AttenuationMapResampleImageFilter->Update();
      rotatedAttenuation = m_AttenuationMapResampleImageFilter->GetOutput();
      rotatedAttenuation->DisconnectPipeline();
      m_AttenuationMapRegionOfInterest->SetInput(rotatedAttenuation);
      m_AttenuationMapRegionOfInterest->SetRegionOfInterest(desiredRegion);
      m_AttenuationMapRegionOfInterest->UpdateOutputInformation();
      m_AttenuationMapMultiplyImageFilter->SetInput1(m_RegionOfInterest->GetOutput());
      m_AttenuationMapMultiplyImageFilter->SetInput2(m_AttenuationMapExpImageFilter->GetOutput());
      m_AttenuationMapMultiplyImageFilter->Update();
      m_AttenuationMapChangeInformation->SetInput(m_AttenuationMapExpImageFilter->GetOutput());
      currentSlice = m_AttenuationMapMultiplyImageFilter->GetOutput();
    }
    currentSlice->DisconnectPipeline();
    m_DiscreteGaussianFilter->SetInput(currentSlice);
    m_ChangeInformation->SetInput(m_RegionOfInterest->GetOutput());

    // Compute the distance between the current slice and the detector
    rotatedVolume->TransformIndexToPhysicalPoint(indexSlice, pointSlice);
    dist = geometry->GetSourceToIsocenterDistances()[nbProjections] -
           pointSlice.GetVectorFromOrigin() * m_VectorOrthogonalDetector;

    unsigned int index;
    for (index = 1; index < nbSlice; index++)
    {
      if (dist - rotatedVolume->GetSpacing()[2] < 0)
      {
        break;
      }
      // Compute the variance of the PSF for the current slice
      sigmaSlice = dist * 2 * thicknessSlice * pow(m_Alpha, 2.0) + 2 * thicknessSlice * m_Alpha * m_SigmaZero -
                   pow(m_Alpha, 2.0) * pow(thicknessSlice, 2.0);
      m_DiscreteGaussianFilter->SetVariance(sigmaSlice);

      // Extract the next slice
      desiredRegion.SetIndex(Dimension - 1, index);
      m_RegionOfInterest->SetRegionOfInterest(desiredRegion);
      m_RegionOfInterest->UpdateOutputInformation();

      // Add the current blur slice with the next
      m_AddImageFilter->GetOutput()->UpdateOutputInformation();
      m_AddImageFilter->GetOutput()->PropagateRequestedRegion();
      m_AddImageFilter->Update();
      if (!(this->GetInput(2)))
      {
        currentSlice = m_AddImageFilter->GetOutput();
      }
      else
      {
        m_AttenuationMapRegionOfInterest->SetRegionOfInterest(desiredRegion);
        m_AttenuationMapRegionOfInterest->UpdateOutputInformation();
        m_AttenuationMapMultiplyImageFilter->SetInput1(m_AddImageFilter->GetOutput());
        m_AttenuationMapMultiplyImageFilter->SetInput2(m_AttenuationMapChangeInformation->GetOutput());
        m_AttenuationMapMultiplyImageFilter->Update();
        currentSlice = m_AttenuationMapMultiplyImageFilter->GetOutput();
      }
      currentSlice->DisconnectPipeline();
      m_DiscreteGaussianFilter->SetInput(currentSlice);
      dist -= rotatedVolume->GetSpacing()[2];
    }
    // Compute the variance of the PSF for the last slice
    sigmaSlice = pow(m_Alpha * dist + m_SigmaZero, 2.0);
    m_DiscreteGaussianFilter->SetVariance(sigmaSlice);

    // Paste the projection in the output volume
    indexProjection[Dimension - 1] = nbProjections + projRegion.GetIndex(Dimension - 1);
    m_PasteImageFilter->SetSourceRegion(m_DiscreteGaussianFilter->GetOutput()->GetLargestPossibleRegion());
    m_PasteImageFilter->SetDestinationIndex(indexProjection);
    m_PasteImageFilter->UpdateLargestPossibleRegion();
    pimg = m_PasteImageFilter->GetOutput();

    pimg->DisconnectPipeline();
    m_PasteImageFilter->SetDestinationImage(pimg);
    nbProjections++;
  }
  m_MultiplyImageFilter->SetInput(pimg);
  m_MultiplyImageFilter->UpdateLargestPossibleRegion();
  pimg = m_MultiplyImageFilter->GetOutput();
  pimg->DisconnectPipeline();
  this->GetOutput()->SetPixelContainer(pimg->GetPixelContainer());
  this->GetOutput()->CopyInformation(pimg);
  this->GetOutput()->SetBufferedRegion(pimg->GetBufferedRegion());
  this->GetOutput()->SetRequestedRegion(pimg->GetRequestedRegion());
}

} // end namespace rtk

#endif
