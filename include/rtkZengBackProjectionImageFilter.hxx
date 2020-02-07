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

#ifndef rtkZengBackProjectionImageFilter_hxx
#define rtkZengBackProjectionImageFilter_hxx

#include <math.h>

#include "rtkZengBackProjectionImageFilter.h"

#include "rtkHomogeneousMatrix.h"
#include "rtkBoxShape.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"
#include <itkImageFileWriter.h>

#include <itkImageRegionIteratorWithIndex.h>
#include <itkInputDataObjectConstIterator.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
ZengBackProjectionImageFilter<TInputImage, TOutputImage>::ZengBackProjectionImageFilter()
{
  // Set default parameters
  m_SigmaZero = 1.5417233052142099;
  m_Alpha = 0.016241189545787734;
  m_VectorOrthogonalDetector[0] = 0;
  m_VectorOrthogonalDetector[1] = 0;
  m_VectorOrthogonalDetector[2] = 1;
  m_centerVolume.Fill(0);

  // Create each filter of the composite filter
  m_AddImageFilter = AddImageFilterType::New();
  m_PasteImageFilter = PasteImageFilterType::New();
  m_DiscreteGaussianFilter = DiscreteGaussianFilterType::New();
  m_ResampleImageFilter = ResampleImageFilterType::New();
  m_Transform = TransformType::New();
  m_MultiplyImageFilter = MultiplyImageFilterType::New();
  m_ExtractImageFilter = ExtractImageFilterType::New();
  m_ConstantVolumeSource = ConstantVolumeSourceType::New();
  m_AttenuationMapExpImageFilter = nullptr;
  m_AttenuationMapMultiplyImageFilter = nullptr;
  m_AttenuationMapRegionOfInterest = nullptr;
  m_AttenuationMapResampleImageFilter = nullptr;
  m_AttenuationMapConstantMultiplyImageFilter = nullptr;
  m_AttenuationMapChangeInformation = nullptr;
  m_AttenuationMapTransform = nullptr;

  // Permanent internal connections
  m_AddImageFilter->SetInput2(m_MultiplyImageFilter->GetOutput());
  m_MultiplyImageFilter->SetInput(m_ResampleImageFilter->GetOutput());

  // Default parameters
  m_DiscreteGaussianFilter->SetMaximumError(0.00001);
  m_DiscreteGaussianFilter->SetFilterDimensionality(2);
#if (ITK_VERSION_MAJOR == 5) && (ITK_VERSION_MINOR >= 1)
  m_DiscreteGaussianFilter->SetInputBoundaryCondition(&m_BoundsCondition);
  m_DiscreteGaussianFilter->SetRealBoundaryCondition(&m_BoundsCondition);
#endif
}

template <class TInputImage, class TOutputImage>
void
ZengBackProjectionImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();
  // Input 2 is the attenuation map relative to the volume
  typename Superclass::InputImagePointer inputPtr2 = const_cast<TInputImage *>(this->GetInput(2));
  if (!inputPtr2)
    return;
}

template <class TInputImage, class TOutputImage>
void
ZengBackProjectionImageFilter<TInputImage, TOutputImage>::VerifyInputInformation() const
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
    if (it.GetName() != "_1")
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
    if (it.GetName() != "_1")
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
ZengBackProjectionImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{

  // Info of the input volume
  const typename InputCPUImageType::PointType   originVolume = this->GetInput(0)->GetOrigin();
  const typename InputCPUImageType::SpacingType spacingVolume = this->GetInput(0)->GetSpacing();
  const typename InputCPUImageType::SizeType    sizeVolume = this->GetInput(0)->GetLargestPossibleRegion().GetSize();

  // Info of the input stack of projections
  const typename OuputCPUImageType::SpacingType spacingProjections = this->GetInput(1)->GetSpacing();
  const typename OuputCPUImageType::SizeType  sizeProjection = this->GetInput(1)->GetLargestPossibleRegion().GetSize();
  const typename OuputCPUImageType::PointType originProjection = this->GetInput(1)->GetOrigin();

  // Find the center of the volume
  m_centerVolume[0] = originVolume[0] + spacingVolume[0] * (double)(sizeVolume[0] - 1) / 2.0;
  m_centerVolume[1] = originVolume[1] + spacingVolume[1] * (double)(sizeVolume[1] - 1) / 2.0;
  m_centerVolume[2] = originVolume[2] + spacingVolume[2] * (double)(sizeVolume[2] - 1) / 2.0;

  PointType centerRotation;
  centerRotation.Fill(0);
  m_Transform->SetCenter(centerRotation);
  m_ResampleImageFilter->SetTransform(m_Transform);
  m_ResampleImageFilter->SetOutputParametersFromImage(this->GetInput(0));

  // Set the output size of volume
  typename InputCPUImageType::SizeType outputSize;
  outputSize[0] = sizeProjection[0];
  outputSize[1] = sizeProjection[1];
  outputSize[2] = sizeVolume[2] * std::sqrt(2);

  // Set the new origin of the volume
  typename InputCPUImageType::PointType outputOrigin;
  outputOrigin[0] = originProjection[0];
  outputOrigin[1] = originProjection[1];
  outputOrigin[2] = m_centerVolume[2] - spacingVolume[2] * (double)(outputSize[2] - 1) / 2.0;

  // Set the output spacing of the volume
  typename InputCPUImageType::SpacingType outputSpacing;
  outputSpacing[0] = spacingProjections[0];
  outputSpacing[1] = spacingProjections[1];
  outputSpacing[2] = spacingVolume[2];

  // We only set the first sub-stack at that point, the rest will be
  // requested in the GenerateData function
  typename ExtractImageFilterType::InputImageRegionType projRegion;
  const unsigned int                                    Dimension = this->InputImageDimension;
  projRegion = this->GetInput(1)->GetLargestPossibleRegion();
  m_ExtractImageFilter->SetExtractionRegion(projRegion);
  m_ExtractImageFilter->SetInput(this->GetInput(1));
  m_ExtractImageFilter->UpdateOutputInformation();

  m_DiscreteGaussianFilter->SetVariance(pow(m_SigmaZero, 2.0));

  if (!(this->GetInput(2)))
  {
    m_DiscreteGaussianFilter->SetInput(m_ExtractImageFilter->GetOutput());
  }
  else
  {
    m_AttenuationMapExpImageFilter = ExpImageFilterType::New();
    m_AttenuationMapMultiplyImageFilter = MultiplyImageFilterType::New();
    m_AttenuationMapRegionOfInterest = RegionOfInterestFilterType::New();
    m_AttenuationMapResampleImageFilter = ResampleImageFilterType::New();
    m_AttenuationMapConstantMultiplyImageFilter = MultiplyImageFilterType::New();
    m_AttenuationMapChangeInformation = ChangeInformationFilterType::New();
    m_AttenuationMapTransform = TransformType::New();
    m_AttenuationMapChangeInformation->ChangeOriginOn();
    m_AttenuationMapChangeInformation->ChangeRegionOn();
    m_AttenuationMapChangeInformation->SetReferenceImage(m_ExtractImageFilter->GetOutput());
    m_AttenuationMapChangeInformation->SetUseReferenceImage(true);

    m_AttenuationMapTransform->SetCenter(centerRotation);
    m_AttenuationMapResampleImageFilter->SetTransform(m_AttenuationMapTransform);
    m_AttenuationMapResampleImageFilter->SetOutputParametersFromImage(this->GetInput(0));

    m_AttenuationMapResampleImageFilter->SetSize(outputSize);
    m_AttenuationMapResampleImageFilter->SetOutputOrigin(outputOrigin);
    m_AttenuationMapResampleImageFilter->SetOutputSpacing(spacingProjections);
    m_AttenuationMapResampleImageFilter->SetInput(this->GetInput(2));
    m_AttenuationMapResampleImageFilter->UpdateOutputInformation();

    typename RegionOfInterestFilterType::InputImageRegionType attRegion;
    attRegion = m_AttenuationMapResampleImageFilter->GetOutput()->GetLargestPossibleRegion();
    attRegion.SetSize(Dimension - 1, 1);
    m_AttenuationMapRegionOfInterest->SetRegionOfInterest(attRegion);
    m_AttenuationMapRegionOfInterest->SetInput(m_AttenuationMapResampleImageFilter->GetOutput());
    m_AttenuationMapRegionOfInterest->UpdateOutputInformation();

    m_AttenuationMapConstantMultiplyImageFilter->SetInput(m_AttenuationMapRegionOfInterest->GetOutput());
    m_AttenuationMapConstantMultiplyImageFilter->SetConstant(-spacingVolume[2]);
    m_AttenuationMapExpImageFilter->SetInput(m_AttenuationMapConstantMultiplyImageFilter->GetOutput());
    m_AttenuationMapChangeInformation->SetInput(m_AttenuationMapExpImageFilter->GetOutput());
    m_AttenuationMapMultiplyImageFilter->SetInput1(m_ExtractImageFilter->GetOutput());
    m_AttenuationMapMultiplyImageFilter->SetInput2(m_AttenuationMapChangeInformation->GetOutput());
    m_DiscreteGaussianFilter->SetInput(m_AttenuationMapMultiplyImageFilter->GetOutput());
  }

  m_MultiplyImageFilter->SetConstant(spacingVolume[2]);

  m_ConstantVolumeSource->SetInformationFromImage(const_cast<TInputImage *>(this->GetInput(0)));
  m_ConstantVolumeSource->SetSpacing(outputSpacing);
  m_ConstantVolumeSource->SetOrigin(outputOrigin);
  m_ConstantVolumeSource->SetSize(outputSize);
  m_ConstantVolumeSource->SetConstant(0);

  m_PasteImageFilter->SetSourceImage(m_DiscreteGaussianFilter->GetOutput());
  m_PasteImageFilter->SetDestinationImage(m_ConstantVolumeSource->GetOutput());
  m_PasteImageFilter->SetSourceRegion(m_DiscreteGaussianFilter->GetOutput()->GetLargestPossibleRegion());

  m_ResampleImageFilter->SetInput(m_PasteImageFilter->GetOutput());
  m_AddImageFilter->SetInput1(this->GetInput(0));

  // Update output information
  m_AddImageFilter->UpdateOutputInformation();
  this->GetOutput()->SetOrigin(m_AddImageFilter->GetOutput()->GetOrigin());
  this->GetOutput()->SetSpacing(m_AddImageFilter->GetOutput()->GetSpacing());
  this->GetOutput()->SetDirection(m_AddImageFilter->GetOutput()->GetDirection());
  this->GetOutput()->SetLargestPossibleRegion(m_AddImageFilter->GetOutput()->GetLargestPossibleRegion());
}

template <class TInputImage, class TOutputImage>
void
ZengBackProjectionImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  const typename Superclass::GeometryType::ConstPointer geometry = this->GetGeometry();
  const unsigned int                                    Dimension = this->InputImageDimension;

  typename ExtractImageFilterType::InputImageRegionType projRegion;
  projRegion = this->GetInput(1)->GetLargestPossibleRegion();
  int                 indexProj = 0;
  std::vector<double> list_angle;
  if (geometry->GetGantryAngles().size() != projRegion.GetSize(Dimension - 1))
  {
    indexProj = projRegion.GetIndex(Dimension - 1);
    list_angle.push_back(geometry->GetGantryAngles()[indexProj]);
  }
  else
  {
    list_angle = geometry->GetGantryAngles();
  }

  projRegion.SetSize(Dimension - 1, 1);

  typename OuputCPUImageType::Pointer                       currentSlice;
  typename OuputCPUImageType::Pointer                       pimg;
  typename OuputCPUImageType::Pointer                       currentVolume;
  typename OuputCPUImageType::PointType                     pointSlice;
  typename OuputCPUImageType::IndexType                     indexSlice;
  typename OuputCPUImageType::Pointer                       rotatedAttenuation;
  typename RegionOfInterestFilterType::InputImageRegionType desiredRegion;
  PointType                                                 centerRotatedVolume;
  PointType                                                 originRotatedVolume;

  indexSlice.Fill(0);
  float dist, sigmaSlice;
  float thicknessSlice = this->GetInput(0)->GetSpacing()[2];
  int   nbProjections = 0;
  int   startSlice;
  for (auto & angle : list_angle)
  {
    // Get the center of the rotated volume
    m_Transform->SetRotation(0., angle, 0.);
    centerRotatedVolume = m_Transform->GetMatrix() * m_centerVolume;

    // Set the new origin of the rotate volume according to the center
    originRotatedVolume = m_ConstantVolumeSource->GetOrigin();
    originRotatedVolume[2] = centerRotatedVolume[2] - m_ConstantVolumeSource->GetSpacing()[2] *
                                                        (double)(m_ConstantVolumeSource->GetSize()[2] - 1) / 2.0;
    m_ConstantVolumeSource->SetOrigin(originRotatedVolume);

    // Set the rotation angle.
    m_Transform->SetRotation(0., -angle, 0.);

    // Extract the projection corresponding to the current angle from the projection stack
    projRegion.SetIndex(Dimension - 1, nbProjections + indexProj);
    m_ExtractImageFilter->SetExtractionRegion(projRegion);
    m_ExtractImageFilter->UpdateOutputInformation();

    // Find the first positive distance between the volume and the detector
    m_ConstantVolumeSource->Update();
    startSlice = m_ConstantVolumeSource->GetOutput()->GetLargestPossibleRegion().GetSize()[2];
    dist = -1;
    while (dist < 0)
    {
      startSlice -= 1;
      indexSlice[Dimension - 1] = startSlice;
      m_ConstantVolumeSource->GetOutput()->TransformIndexToPhysicalPoint(indexSlice, pointSlice);
      dist = geometry->GetSourceToIsocenterDistances()[nbProjections] -
             pointSlice.GetVectorFromOrigin() * m_VectorOrthogonalDetector;
    }
    if (this->GetInput(2))
    {
      m_AttenuationMapTransform->SetRotation(0., angle, 0.);
      m_AttenuationMapResampleImageFilter->SetOutputOrigin(originRotatedVolume);
      m_AttenuationMapResampleImageFilter->Update();
      rotatedAttenuation = m_AttenuationMapResampleImageFilter->GetOutput();
      rotatedAttenuation->DisconnectPipeline();
      desiredRegion = rotatedAttenuation->GetLargestPossibleRegion();
      desiredRegion.SetSize(Dimension - 1, 1);
      desiredRegion.SetIndex(Dimension - 1, startSlice);
      m_AttenuationMapRegionOfInterest->SetInput(rotatedAttenuation);
      m_AttenuationMapRegionOfInterest->SetRegionOfInterest(desiredRegion);
      m_AttenuationMapRegionOfInterest->UpdateOutputInformation();
    }
    // Compute the variance of the PSF for the first slice
    sigmaSlice = pow(m_Alpha * dist + m_SigmaZero, 2.0);
    m_DiscreteGaussianFilter->SetVariance(sigmaSlice);
    m_DiscreteGaussianFilter->UpdateLargestPossibleRegion();
    currentSlice = m_DiscreteGaussianFilter->GetOutput();
    currentSlice->DisconnectPipeline();

    // Paste the blur slice into the output volume
    m_PasteImageFilter->SetSourceImage(currentSlice);
    m_PasteImageFilter->SetSourceRegion(currentSlice->GetLargestPossibleRegion());
    m_PasteImageFilter->SetDestinationIndex(indexSlice);
    m_PasteImageFilter->Update();
    currentVolume = m_PasteImageFilter->GetOutput();
    currentVolume->DisconnectPipeline();
    m_PasteImageFilter->SetDestinationImage(currentVolume);
    if (!this->GetInput(2))
    {
      m_DiscreteGaussianFilter->SetInput(currentSlice);
    }
    else
    {
      desiredRegion.SetIndex(Dimension - 1, startSlice - 1);
      m_AttenuationMapRegionOfInterest->SetRegionOfInterest(desiredRegion);
      m_AttenuationMapRegionOfInterest->UpdateOutputInformation();
      m_AttenuationMapMultiplyImageFilter->SetInput1(currentSlice);
      m_DiscreteGaussianFilter->SetInput(m_AttenuationMapMultiplyImageFilter->GetOutput());
    }
    int index;
    for (index = startSlice; index > 0; index--)
    {
      // Compute the distance between the current slice and the detector
      indexSlice[Dimension - 1] = index - 1;
      dist += m_ConstantVolumeSource->GetSpacing()[2];
      // Compute the variance of the PSF for the current slice
      sigmaSlice = dist * 2 * thicknessSlice * pow(m_Alpha, 2.0) + 2 * thicknessSlice * m_Alpha * m_SigmaZero -
                   pow(m_Alpha, 2.0) * pow(thicknessSlice, 2.0);
      m_DiscreteGaussianFilter->SetVariance(sigmaSlice);
      m_DiscreteGaussianFilter->Update();
      currentSlice = m_DiscreteGaussianFilter->GetOutput();
      currentSlice->DisconnectPipeline();

      // Paste the blur slice into the output volume
      m_PasteImageFilter->SetSourceImage(currentSlice);
      m_PasteImageFilter->SetSourceRegion(currentSlice->GetLargestPossibleRegion());
      m_PasteImageFilter->SetDestinationIndex(indexSlice);
      m_PasteImageFilter->Update();
      currentVolume = m_PasteImageFilter->GetOutput();
      currentVolume->DisconnectPipeline();
      m_PasteImageFilter->SetDestinationImage(currentVolume);
      if (!this->GetInput(2))
      {
        m_DiscreteGaussianFilter->SetInput(currentSlice);
      }
      else
      {
        desiredRegion.SetIndex(Dimension - 1, index - 1);
        m_AttenuationMapRegionOfInterest->SetRegionOfInterest(desiredRegion);
        m_AttenuationMapRegionOfInterest->UpdateOutputInformation();
        m_AttenuationMapMultiplyImageFilter->SetInput1(currentSlice);
        m_DiscreteGaussianFilter->SetInput(m_AttenuationMapMultiplyImageFilter->GetOutput());
      }
    }
    // Rotate the volume
    m_ResampleImageFilter->SetInput(currentVolume);
    m_AddImageFilter->Update();
    pimg = m_AddImageFilter->GetOutput();
    pimg->DisconnectPipeline();
    m_AddImageFilter->SetInput1(pimg);
    if (!this->GetInput(2))
    {
      m_DiscreteGaussianFilter->SetInput(m_ExtractImageFilter->GetOutput());
    }
    else
    {
      m_AttenuationMapMultiplyImageFilter->SetInput1(m_ExtractImageFilter->GetOutput());
      m_DiscreteGaussianFilter->SetInput(m_AttenuationMapMultiplyImageFilter->GetOutput());
    }
    nbProjections++;
  }
  this->GetOutput()->SetPixelContainer(pimg->GetPixelContainer());
  this->GetOutput()->CopyInformation(pimg);
  this->GetOutput()->SetBufferedRegion(pimg->GetBufferedRegion());
  this->GetOutput()->SetRequestedRegion(pimg->GetRequestedRegion());
}

} // end namespace rtk

#endif
