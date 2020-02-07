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

#ifndef rtkJosephBackAttenuatedProjectionImageFilter_hxx
#define rtkJosephBackAttenuatedProjectionImageFilter_hxx

#include "rtkJosephBackAttenuatedProjectionImageFilter.h"

#include "rtkHomogeneousMatrix.h"
#include "rtkBoxShape.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionIteratorWithIndex.h>
#include <itkInputDataObjectConstIterator.h>

namespace rtk
{
template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
JosephBackAttenuatedProjectionImageFilter<TInputImage,
                                          TOutputImage,
                                          TInterpolationWeightMultiplication,
                                          TSplatWeightMultiplication,
                                          TSumAlongRay>::JosephBackAttenuatedProjectionImageFilter()
{
  this->m_InferiorClip = 0.;
  this->m_SuperiorClip = 1.;
  this->SetNumberOfRequiredInputs(3);
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
void
JosephBackAttenuatedProjectionImageFilter<TInputImage,
                                          TOutputImage,
                                          TInterpolationWeightMultiplication,
                                          TSplatWeightMultiplication,
                                          TSumAlongRay>::GenerateInputRequestedRegion()
{
  // Input 2 is the attenuation map relative to the volume
  typename Superclass::InputImagePointer inputPtr2 = const_cast<TInputImage *>(this->GetInput(2));
  if (!inputPtr2)
    return;
  Superclass::GenerateInputRequestedRegion();
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
void
JosephBackAttenuatedProjectionImageFilter<TInputImage,
                                          TOutputImage,
                                          TInterpolationWeightMultiplication,
                                          TSplatWeightMultiplication,
                                          TSumAlongRay>::VerifyInputInformation() const
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

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
void
JosephBackAttenuatedProjectionImageFilter<TInputImage,
                                          TOutputImage,
                                          TInterpolationWeightMultiplication,
                                          TSplatWeightMultiplication,
                                          TSumAlongRay>::Init()
{
  this->m_InterpolationWeightMultiplication.SetAttenuationMinusEmissionMapsPtrDiff(
    this->GetInput(2)->GetBufferPointer() - this->GetInput(0)->GetBufferPointer());
  this->m_SumAlongRay.SetAttenuationPixel(this->m_InterpolationWeightMultiplication.GetAttenuationPixel());
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
void
JosephBackAttenuatedProjectionImageFilter<TInputImage,
                                          TOutputImage,
                                          TInterpolationWeightMultiplication,
                                          TSplatWeightMultiplication,
                                          TSumAlongRay>::GenerateData()
{
  Init();
  Superclass::GenerateData();
}
} // end namespace rtk

#endif
