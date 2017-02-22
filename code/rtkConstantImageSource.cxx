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

#include "rtkConstantImageSource.h"

namespace rtk
{

template <>
itk::VariableLengthVector<float>
rtk::ConstantImageSource<itk::VectorImage<float, 3> >
::FillPixel(float value)
{
  itk::VariableLengthVector<float> vect;
  vect.SetSize(this->GetVectorLength());
  vect.Fill(value);
  return (vect);
}

} // end namespace rtk
