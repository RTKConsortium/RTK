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

#ifndef rtkSheppLoganPhantomFilter_hxx
#define rtkSheppLoganPhantomFilter_hxx

#include "rtkSheppLoganPhantom.h"

namespace rtk
{
template <class TInputImage, class TOutputImage>
SheppLoganPhantomFilter<TInputImage, TOutputImage>
::SheppLoganPhantomFilter()
{
  this->SetPhantomScale(ConvexObject::VectorType(128));
}

template <class TInputImage, class TOutputImage>
void
SheppLoganPhantomFilter<TInputImage, TOutputImage>
::GenerateData()
{
  this->SetGeometricPhantom( SheppLoganPhantom::New().GetPointer() );
  this->SetConfigFile("");
  Superclass::GenerateData();
}
} // end namespace rtk

#endif
