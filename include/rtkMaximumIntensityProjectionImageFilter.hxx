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
  auto sumAlongRayFunc =
    [](const ThreadIdType, OutputPixelType & mipValue, const InputPixelType volumeValue, const VectorType &) {
      OutputPixelType tmp = static_cast<OutputPixelType>(volumeValue);
      if (tmp > mipValue)
      {
        mipValue = tmp;
      }
    };
  this->SetSumAlongRay(sumAlongRayFunc);

  auto projectedValueAccumulationFunc = [](const ThreadIdType,
                                           const InputPixelType &  input,
                                           OutputPixelType &       output,
                                           const OutputPixelType & rayCastValue,
                                           const VectorType &      stepInMM,
                                           const VectorType &,
                                           const VectorType &,
                                           const VectorType &,
                                           const VectorType &) {
    OutputPixelType tmp = static_cast<OutputPixelType>(input);
    if (tmp < rayCastValue)
    {
      tmp = rayCastValue;
    }
    output = tmp * stepInMM.GetNorm();
  };
  this->SetProjectedValueAccumulation(projectedValueAccumulationFunc);
}

} // end namespace rtk

#endif
