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

#ifndef rtkJosephBackAttenuatedProjectionImageFilter_hxx
#define rtkJosephBackAttenuatedProjectionImageFilter_hxx


#include "rtkHomogeneousMatrix.h"
#include "rtkBoxShape.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionIteratorWithIndex.h>
#include <itkInputDataObjectConstIterator.h>

namespace rtk
{
template <class TInputImage, class TOutputImage>
JosephBackAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::JosephBackAttenuatedProjectionImageFilter()
  : JosephBackProjectionImageFilter<TInputImage, TOutputImage>()
{
  this->m_InferiorClip = 0.;
  this->m_SuperiorClip = 1.;
  this->SetNumberOfRequiredInputs(3);

  this->SetStepFunction(
    [this](OutputPixelType * volume, AttenuationRayState & ray, const typename Superclass::StepSamples & samples) {
      const auto      attenuation = this->SampleAttenuation(samples.offsets);
      InputPixelType  stepAttenuation = 0;
      OutputPixelType interpolatedAttenuation{};
      for (unsigned int k = 0; k < 4; ++k)
      {
        const double w = static_cast<WeightCoordinateType>(samples.weights[k]) * samples.stepLengthInVoxel;
        stepAttenuation += w * attenuation[k];
        interpolatedAttenuation += static_cast<OutputPixelType>(w * attenuation[k]);
      }
      ray.attenuationRay += interpolatedAttenuation;
      const auto           ex2 = static_cast<InputPixelType>(exp(-ray.attenuationRay * samples.stepSizeInMM));
      const InputPixelType wf =
        stepAttenuation > 0 ? (ray.ex1 - ex2) / stepAttenuation : ray.ex1 * samples.stepSizeInMM;
      ray.ex1 = ex2;
      const auto value = static_cast<OutputPixelType>(wf * samples.projectionValue);
      for (unsigned int k = 0; k < 4; ++k)
      {
        volume[samples.offsets[k]] +=
          static_cast<OutputPixelType>(value * samples.weights[k] * samples.stepLengthInVoxel);
      }
    },
    AttenuationRayState());

  this->DynamicMultiThreadingOff();
}

template <class TInputImage, class TOutputImage>
void
JosephBackAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  // Input 2 is the attenuation map relative to the volume
  typename Superclass::InputImagePointer inputPtr2 = const_cast<TInputImage *>(this->GetInput(2));
  if (!inputPtr2)
    return;
  Superclass::GenerateInputRequestedRegion();
}

template <class TInputImage, class TOutputImage>
void
JosephBackAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::VerifyInputInformation() const
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
JosephBackAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::Init()
{
  const TInputImage * attenuationMap = this->GetInput(2);
  if (!attenuationMap)
  {
    itkExceptionMacro("Attenuation map (input 2) must be set before running the attenuated back projector.");
  }
  if (!attenuationMap->GetBufferedRegion().IsInside(this->GetOutput()->GetRequestedRegion()))
  {
    itkExceptionMacro("The attenuation map (input 2) must cover the requested region of the output volume.");
  }
  m_AttenuationBuffer = attenuationMap->GetBufferPointer();
  m_AttenuationSameLayout = attenuationMap->GetBufferedRegion() == this->GetOutput()->GetBufferedRegion();
}

template <class TInputImage, class TOutputImage>
void
JosephBackAttenuatedProjectionImageFilter<TInputImage, TOutputImage>::AllocateOutputs()
{
  Superclass::AllocateOutputs();
  Init();
}

} // end namespace rtk

#endif
