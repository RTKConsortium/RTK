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

#include "srtkInterpolator.h"

#define srtkInterpolatorToStringCaseMacro(n) case srtk##n: return ( os << #n )

namespace rtk
{
namespace simple
{

std::ostream& operator<<(std::ostream& os, const InterpolatorEnum i)
{
  switch (i)
    {
    srtkInterpolatorToStringCaseMacro(NearestNeighbor);
    srtkInterpolatorToStringCaseMacro(Linear);
    srtkInterpolatorToStringCaseMacro(BSpline);
    srtkInterpolatorToStringCaseMacro(Gaussian);
    srtkInterpolatorToStringCaseMacro(LabelGaussian);
    srtkInterpolatorToStringCaseMacro(HammingWindowedSinc);
    srtkInterpolatorToStringCaseMacro(CosineWindowedSinc);
    srtkInterpolatorToStringCaseMacro(WelchWindowedSinc);
    srtkInterpolatorToStringCaseMacro(LanczosWindowedSinc);
    srtkInterpolatorToStringCaseMacro(BlackmanWindowedSinc);
    }
  return os;
}

} // end namespace simple
} // end namespace rtk
