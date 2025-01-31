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

#include "rtkHncImageIOFactory.h"

#include <fstream>

//====================================================================
rtk::HncImageIOFactory::HncImageIOFactory()
{
  this->RegisterOverride(
    "itkImageIOBase", "HncImageIO", "Hnc Image IO", 1, itk::CreateObjectFunction<HncImageIO>::New());
}

// Undocumented API used to register during static initialization.
// DO NOT CALL DIRECTLY.

namespace itk
{

static bool HncImageIOFactoryHasBeenRegistered;

void RTK_EXPORT
HncImageIOFactoryRegister__Private()
{
  if (!HncImageIOFactoryHasBeenRegistered)
  {
    HncImageIOFactoryHasBeenRegistered = true;
    rtk::HncImageIOFactory::RegisterOneFactory();
  }
}

} // end namespace itk
