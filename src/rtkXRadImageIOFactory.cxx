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

#include "rtkXRadImageIOFactory.h"

//====================================================================
rtk::XRadImageIOFactory::XRadImageIOFactory()
{
  this->RegisterOverride(
    "itkImageIOBase", "XRadImageIO", "XRad Image IO", true, itk::CreateObjectFunction<XRadImageIO>::New());
}

// Undocumented API used to register during static initialization.
// DO NOT CALL DIRECTLY.

namespace itk
{

static bool XRadImageIOFactoryHasBeenRegistered;

void RTK_EXPORT
XRadImageIOFactoryRegister__Private()
{
  if (!XRadImageIOFactoryHasBeenRegistered)
  {
    XRadImageIOFactoryHasBeenRegistered = true;
    rtk::XRadImageIOFactory::RegisterOneFactory();
  }
}

} // end namespace itk
