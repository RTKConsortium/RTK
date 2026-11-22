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

#ifndef rtkJosephForwardAttenuatedProjectionImageFilter_hxx
#define rtkJosephForwardAttenuatedProjectionImageFilter_hxx


#include "rtkHomogeneousMatrix.h"
#include "rtkBoxShape.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>
#include <itkInputDataObjectConstIterator.h>

namespace rtk
{
template <class TInputImage, class TOutputImage>
JosephForwardAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::JosephForwardAttenuatedProjectionImageFilter()
  : JosephForwardProjectionImageFilter<TInputImage, TOutputImage>()
{
  std::fill(m_AttenuationRay.begin(), m_AttenuationRay.end(), 0.);
  std::fill(m_AttenuationPixel.begin(), m_AttenuationPixel.end(), 0.);
  std::fill(m_Ex1.begin(), m_Ex1.end(), 1.);
  m_AttenuationMinusEmissionMapsPtrDiff = 0;

  /** Function to multiply the interpolation weights with the projected
   * volume values and attenuation map. */
  auto interpolationWeightMultiplicationAttenuatedFunc = [](const ThreadIdType         threadId,
                                                            Superclass &               filter,
                                                            double                     stepLengthInVoxel,
                                                            const WeightCoordinateType weight,
                                                            const InputPixelType *     p,
                                                            const ptrdiff_t            i) -> OutputPixelType {
    Self &            self = static_cast<Self &>(filter);
    const std::size_t t = threadId * ThreadStride;
    const double      w = weight * stepLengthInVoxel;

    self.m_AttenuationRay[t] += w * (p + self.m_AttenuationMinusEmissionMapsPtrDiff)[i];
    self.m_AttenuationPixel[t] += w * (p + self.m_AttenuationMinusEmissionMapsPtrDiff)[i];
    return weight * p[i];
  };
  this->SetInterpolationWeightMultiplication(interpolationWeightMultiplicationAttenuatedFunc);

  /** Function to compute the attenuation correction on the projection. */
  auto computeAttenuationCorrectionFunc = [](const ThreadIdType   threadId,
                                             Superclass &         filter,
                                             OutputPixelType &    sumValue,
                                             const InputPixelType volumeValue,
                                             const VectorType &   stepInMM) {
    Self &            self = static_cast<Self &>(filter);
    const std::size_t t = threadId * ThreadStride;
    InputPixelType    ex2 = exp(-1. * self.m_AttenuationRay[t] * stepInMM.GetNorm());
    InputPixelType    wf;

    if (self.m_AttenuationPixel[t] > 0)
    {
      wf = (self.m_Ex1[t] - ex2) / self.m_AttenuationPixel[t];
    }
    else
    {
      wf = self.m_Ex1[t] * stepInMM.GetNorm();
    }

    self.m_Ex1[t] = ex2;
    self.m_AttenuationPixel[t] = 0;
    sumValue += wf * volumeValue;
  };
  this->SetSumAlongRay(computeAttenuationCorrectionFunc);

  /** Function to accumulate the ray casting on the projection. */
  auto projectedValueAccumulationAttenuatedFunc = [](const ThreadIdType      threadId,
                                                     Superclass &            filter,
                                                     const InputPixelType &  input,
                                                     OutputPixelType &       output,
                                                     const OutputPixelType & rayCastValue,
                                                     const VectorType &      itkNotUsed(stepInMM),
                                                     const PointType &       itkNotUsed(source),
                                                     const VectorType &      itkNotUsed(sourceToPixel),
                                                     const PointType &       itkNotUsed(nearestPoint),
                                                     const PointType &       itkNotUsed(farthestPoint)) {
    Self &            self = static_cast<Self &>(filter);
    const std::size_t t = threadId * ThreadStride;
    output = input + rayCastValue;
    self.m_AttenuationRay[t] = 0;
    self.m_Ex1[t] = 1;
  };
  this->SetProjectedValueAccumulation(projectedValueAccumulationAttenuatedFunc);

  this->SetNumberOfRequiredInputs(3);
  this->DynamicMultiThreadingOff();
}

template <class TInputImage, class TOutputImage>
void
JosephForwardAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();
  // Input 2 is the attenuation map relative to the volume
  typename Superclass::InputImagePointer inputPtr2 = const_cast<TInputImage *>(this->GetInput(2));
  if (!inputPtr2)
    return;

  typename TInputImage::RegionType reqRegion2 = inputPtr2->GetLargestPossibleRegion();
  inputPtr2->SetRequestedRegion(reqRegion2);
}

template <class TInputImage, class TOutputImage>
void
JosephForwardAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::VerifyInputInformation() const
{
  Superclass::VerifyInputInformation();
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
    if (it.GetName() == "_1")
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
    if (it.GetName() == "_2")
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
        const double coordinateTol = std::abs(Self::GetGlobalDefaultCoordinateTolerance() *
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
JosephForwardAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  if (!this->GetInput(2))
  {
    itkExceptionMacro("Attenuation map (input 2) must be set before running the attenuated forward projector.");
  }
  this->m_AttenuationMinusEmissionMapsPtrDiff =
    this->GetInput(2)->GetBufferPointer() - this->GetInput(1)->GetBufferPointer();
}
} // end namespace rtk

#endif
