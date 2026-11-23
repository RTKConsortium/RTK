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

#ifndef rtkMaximumIntensityProjectionImageFilter_hxx
#define rtkMaximumIntensityProjectionImageFilter_hxx

namespace rtk
{

template <class TInputImage, class TOutputImage>
MaximumIntensityProjectionImageFilter<TInputImage, TOutputImage>::MaximumIntensityProjectionImageFilter()
  : JosephForwardProjectionImageFilter<TInputImage, TOutputImage>()
{
  this->SetStepFunction([](OutputPixelType & maxValue,
                           typename Superclass::EmptyRayState &,
                           const typename Superclass::StepSamples & samples) {
    const OutputPixelType value = samples.Interpolate();
    if (value > maxValue)
    {
      maxValue = value;
    }
  });
  this->SetProjectedValueAccumulation([](const InputPixelType &  input,
                                         OutputPixelType &       output,
                                         const OutputPixelType & maxValue,
                                         const VectorType &      stepInMM,
                                         const PointType &,
                                         const VectorType &,
                                         const PointType &,
                                         const PointType &) {
    OutputPixelType value = static_cast<OutputPixelType>(input);
    if (value < maxValue)
    {
      value = maxValue;
    }
    output = value * stepInMM.GetNorm();
  });
}

} // end namespace rtk

#endif
